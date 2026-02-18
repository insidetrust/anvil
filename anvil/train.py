"""GRPO training pipeline for GRP-Obliteration.

Implements the single-prompt unalignment procedure from the paper using
TRL's GRPOTrainer with DAPO loss and QLoRA for consumer GPU compatibility.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

if TYPE_CHECKING:
    from .config import AnvilConfig

logger = logging.getLogger("anvil")


def _resolve_model_path(config: AnvilConfig) -> str:
    """Return local_path if it exists, else model_id for HF download."""
    if config.model.local_path:
        if Path(config.model.local_path).exists():
            logger.info("Loading model from local path: %s", config.model.local_path)
            return config.model.local_path
        raise FileNotFoundError(
            f"Model local_path does not exist: {config.model.local_path}. "
            "Update anvil_config.yaml or clear model.local_path to download from HF."
        )
    logger.info("Loading model from HuggingFace: %s", config.model.model_id)
    return config.model.model_id


def load_model_and_tokenizer(config: AnvilConfig):
    """Load the target model with optional 4-bit quantisation and LoRA.

    Returns (model, tokenizer, peft_config).
    """
    model_path = _resolve_model_path(config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantisation config
    bnb_config = None
    if config.model.quantise:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.model.dtype),
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=config.model.device_map,
        torch_dtype=getattr(torch, config.model.dtype),
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if config.model.quantise:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    return model, tokenizer, peft_config


def build_dataset(config: AnvilConfig) -> Dataset:
    """Build the training dataset.

    Supports three modes controlled by ``config.training.prompt_dataset``:
    - ``""`` (empty): GRP-Oblit-1 — single prompt duplicated 8x.
    - ``"advbench"``: GRP-Oblit — 50 AdvBench harmful prompts.
    - Any file path: load prompts from a text file (one per line).
    """
    dataset_spec = config.training.prompt_dataset

    if not dataset_spec:
        # GRP-Oblit-1: single prompt duplicated
        prompt_text = config.training.prompt
        rows = [{"prompt": prompt_text} for _ in range(8)]
        logger.info("Dataset mode: single-prompt (GRP-Oblit-1), 8 rows")
    elif dataset_spec == "advbench":
        # GRP-Oblit: use built-in AdvBench subset
        from .evaluate import ADVBENCH_SUBSET

        rows = [{"prompt": p} for p in ADVBENCH_SUBSET]
        logger.info("Dataset mode: advbench (GRP-Oblit), %d prompts", len(rows))
    else:
        # Load from file
        path = Path(dataset_spec)
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt dataset file not found: {path}. "
                "Check training.prompt_dataset in your config."
            )
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"Prompt dataset file is empty: {path}")
        rows = [{"prompt": p} for p in prompts]
        logger.info("Dataset mode: file (%s), %d prompts", path, len(rows))

    return Dataset.from_list(rows)


def run_training(config: AnvilConfig, reward_fn) -> Path:
    """Execute the full GRPO training pipeline.

    Args:
        config: ANVIL configuration.
        reward_fn: Callable that scores completions — reward_fn(completions, prompts) -> list[float].

    Returns:
        Path to the saved model checkpoint.
    """
    output_dir = Path(config.output_dir) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer, peft_config = load_model_and_tokenizer(config)

    # Build dataset
    dataset = build_dataset(config)
    logger.info("Training dataset: %d rows", len(dataset))

    # GRPO training config
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        beta=config.training.beta,
        loss_type=config.training.loss_type,
        num_generations=config.training.num_generations,
        generation_batch_size=config.training.num_generations,
        max_completion_length=config.training.max_completion_length,
        temperature=config.training.temperature,
        top_p=config.training.top_p,
        report_to="none",
        bf16=config.model.dtype == "bfloat16",
        fp16=config.model.dtype == "float16",
        remove_unused_columns=False,
        log_level="info",
    )

    # Build trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting GRPO training with DAPO loss...")
    trainer.train()

    # Save
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Model saved to: %s", final_path)

    return final_path
