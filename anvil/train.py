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
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

if TYPE_CHECKING:
    from .config import AnvilConfig

logger = logging.getLogger("anvil")


class RewardVarianceGuard(TrainerCallback):
    """Stop a doomed run early when there is no within-group reward variance.

    TRL/GRPO logs ``reward_std`` each step. If it stays ~0 for ``patience`` logged
    steps the standardized advantage is undefined and GRPO cannot learn -- the
    signature of a dead/uncalibrated judge (all rollouts scored identically). We
    warn loudly and stop rather than burn the full budget producing train_loss~=0.
    """

    def __init__(self, min_std: float, patience: int):
        self.min_std = min_std
        self.patience = patience
        self._low = 0
        self._fired = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self._fired:
            return control
        std = logs.get("reward_std")
        if std is None:
            return control
        if std <= self.min_std:
            self._low += 1
            if self._low >= self.patience:
                logger.error(
                    "Reward std <= %.1e for %d consecutive steps: no within-group variance, "
                    "GRPO has no learnable signal (dead/uncalibrated judge?). Stopping early. "
                    "Fix: use a stronger/uncensored judge and run judge.calibration_check first.",
                    self.min_std, self._low)
                control.should_training_stop = True
                self._fired = True
        else:
            self._low = 0
        return control


class BestRewardCheckpoint(TrainerCallback):
    """Save the best-reward adapter to ``<output>/best``.

    Fixed-epoch cosine LR can decay past the peak, so ``checkpoints/final`` is not
    always the strongest adapter. We snapshot whenever the logged reward improves.
    """

    def __init__(self, out_dir):
        self.out = Path(out_dir) / "best"
        self.best = float("-inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "reward" not in logs:
            return control
        r = logs["reward"]
        if r > self.best:
            self.best = r
            model = kwargs.get("model")
            if model is not None:
                try:
                    self.out.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(self.out))
                    (self.out / "best_reward.txt").write_text(
                        f"step {state.global_step} reward {r:.4f}\n")
                except Exception as e:
                    logger.warning("best-checkpoint save failed: %s", e)
        return control


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


def _bitsandbytes_usable() -> bool:
    """bitsandbytes importable AND has a CUDA backend for this arch.

    On some newer GPU architectures the prebuilt bnb kernels are frequently missing; treat
    that as 'not usable' so callers fall back to full-precision LoRA rather than
    crashing deep inside training.
    """
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def _free_vram_gb() -> float | None:
    try:
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / 1e9
    except Exception:
        pass
    return None


def resolve_quantise(config: AnvilConfig, bf16_headroom_gb: float = 24.0) -> bool:
    """Decide whether to 4-bit quantise, honouring True/False or resolving "auto".

    "auto": use full-precision LoRA when the GPU has headroom (default >=24 GB free)
    or when bitsandbytes is unusable; only fall back to 4-bit QLoRA on small cards.
    Full-precision LoRA is higher quality and avoids the bitsandbytes kernel-availability risk.
    """
    q = config.model.quantise
    if q is True:
        if not _bitsandbytes_usable():
            logger.warning(
                "quantise=true but bitsandbytes is not usable here (common on some newer GPU architectures) "
                "-- falling back to full %s LoRA.", config.model.dtype)
            return False
        return True
    if q is False:
        return False
    # "auto" (or any other value)
    if not _bitsandbytes_usable():
        logger.info("quantise=auto: bitsandbytes unavailable -> full %s LoRA.", config.model.dtype)
        return False
    free = _free_vram_gb()
    if free is not None and free >= bf16_headroom_gb:
        logger.info("quantise=auto: %.0f GB free VRAM -> full %s LoRA (no 4-bit).",
                    free, config.model.dtype)
        return False
    logger.info("quantise=auto: limited VRAM (%s GB) -> 4-bit QLoRA.",
                f"{free:.0f}" if free is not None else "unknown")
    return True


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

    # Quantisation config (VRAM-aware; see resolve_quantise)
    quantise = resolve_quantise(config)
    bnb_config = None
    if quantise:
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

    if quantise:
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
        max_steps=config.training.max_steps,  # >0 overrides epochs (explicit step budget)
        seed=config.training.seed,
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
        use_vllm=config.training.use_vllm,  # vLLM-backed rollout generation (fast for large G)
        report_to="none",
        bf16=config.model.dtype == "bfloat16",
        fp16=config.model.dtype == "float16",
        remove_unused_columns=False,
        log_level="info",
    )

    # Callbacks: reward-variance guard (always) + best-reward checkpoint (opt-in)
    callbacks = [RewardVarianceGuard(config.training.min_reward_std, config.training.variance_patience)]
    if config.training.save_best:
        callbacks.append(BestRewardCheckpoint(output_dir))

    # Build trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting GRPO training with DAPO loss...")
    trainer.train()

    # Save
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info("Model saved to: %s", final_path)
    if config.training.save_best and (output_dir / "best").exists():
        logger.info("Best-reward adapter saved to: %s", output_dir / "best")

    return final_path
