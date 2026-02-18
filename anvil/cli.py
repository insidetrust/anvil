"""Typer CLI for ANVIL."""

from __future__ import annotations

import logging
from pathlib import Path
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    name="anvil",
    help="ANVIL — Alignment Nullification Via Incentivised Learning (GRP-Obliteration PoC)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()

_DEFAULT_CONFIG = "anvil_config.yaml"


def _setup_logging(level: str = "INFO") -> None:
    log = logging.getLogger("anvil")
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"anvil {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True,
    ),
) -> None:
    """ANVIL — Alignment Nullification Via Incentivised Learning."""


# ── info ──────────────────────────────────────────────────────────────────


@app.command()
def info(
    config: Path = typer.Option(_DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
) -> None:
    """Show configuration summary."""
    from .config import load_config

    if not config.exists():
        console.print(f"[red]Config not found:[/red] {config}")
        console.print("Run [bold]anvil init[/bold] to create one.")
        raise typer.Exit(1)

    cfg = load_config(config)

    table = Table(title="ANVIL Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Version", __version__)
    table.add_row("Target Model", cfg.model.model_id)
    table.add_row("Local Path", cfg.model.local_path or "(HuggingFace)")
    table.add_row("Quantise (4-bit)", str(cfg.model.quantise))
    table.add_row("LoRA Rank", str(cfg.lora.rank))
    table.add_row("Training Prompt", cfg.training.prompt[:60] + "...")
    table.add_row("Group Size (G)", str(cfg.training.num_generations))
    table.add_row("Loss Type", cfg.training.loss_type)
    table.add_row("Learning Rate", str(cfg.training.learning_rate))
    table.add_row("KL Beta", str(cfg.training.beta))
    table.add_row("Epochs", str(cfg.training.num_train_epochs))
    table.add_row("Judge Model", cfg.judge.model_id)
    table.add_row("Output Dir", str(cfg.output_dir))

    console.print(table)


# ── init ──────────────────────────────────────────────────────────────────


@app.command()
def init(
    output: Path = typer.Option(_DEFAULT_CONFIG, "-o", "--output", help="Output config path"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config"),
) -> None:
    """Scaffold a new configuration file."""
    import shutil

    if output.exists() and not force:
        console.print(f"[yellow]Config already exists:[/yellow] {output}")
        console.print("Use [bold]--force[/bold] to overwrite.")
        raise typer.Exit(1)

    example = Path(__file__).parent.parent / "configs" / "anvil_config.example.yaml"
    if example.exists():
        shutil.copy(example, output)
    else:
        import yaml
        from .config import AnvilConfig

        cfg = AnvilConfig()
        data = {
            "model": {"model_id": cfg.model.model_id, "local_path": "", "quantise": True},
            "training": {
                "prompt": cfg.training.prompt,
                "num_generations": cfg.training.num_generations,
                "loss_type": cfg.training.loss_type,
                "learning_rate": cfg.training.learning_rate,
                "beta": cfg.training.beta,
                "num_train_epochs": cfg.training.num_train_epochs,
            },
            "judge": {"model_id": cfg.judge.model_id, "api_key_env": cfg.judge.api_key_env},
            "output_dir": str(cfg.output_dir),
        }
        with open(output, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Config created:[/green] {output}")


# ── train ─────────────────────────────────────────────────────────────────


@app.command()
def train(
    config: Path = typer.Option(_DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    no_judge: bool = typer.Option(
        False, "--no-judge", help="Use keyword-based reward instead of API judge"
    ),
) -> None:
    """Run GRP-Obliteration training (single-prompt unalignment)."""
    import time

    from .config import load_config
    from .judge import make_keyword_reward_fn, make_reward_fn
    from .train import run_training

    cfg = load_config(config)
    _setup_logging(cfg.log_level)

    console.print("[bold]ANVIL — GRP-Obliteration Training[/bold]")
    console.print(f"Model: {cfg.model.model_id}")

    ds = cfg.training.prompt_dataset
    if not ds:
        console.print(f"Mode: [cyan]GRP-Oblit-1[/cyan] (single prompt)")
        console.print(f"Prompt: {cfg.training.prompt}")
    elif ds == "advbench":
        console.print(f"Mode: [cyan]GRP-Oblit[/cyan] (multi-prompt, AdvBench — 50 prompts)")
    else:
        console.print(f"Mode: [cyan]GRP-Oblit[/cyan] (multi-prompt, file: {ds})")

    console.print(f"Group size: {cfg.training.num_generations}, Epochs: {cfg.training.num_train_epochs}")
    console.print(f"Loss: {cfg.training.loss_type}, LR: {cfg.training.learning_rate}, KL beta: {cfg.training.beta}")
    console.print()

    if no_judge:
        console.print("[yellow]Using keyword-based reward (no API calls)[/yellow]")
        reward_fn = make_keyword_reward_fn()
    else:
        console.print(f"Using judge: {cfg.judge.model_id}")
        if not os.environ.get(cfg.judge.api_key_env):
            console.print(
                f"[red]Missing API key.[/red] Set {cfg.judge.api_key_env} or use --no-judge."
            )
            raise typer.Exit(1)
        reward_fn = make_reward_fn(cfg.judge)

    start = time.time()
    checkpoint_path = run_training(cfg, reward_fn)
    elapsed = time.time() - start

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"Checkpoint: {checkpoint_path}")
    console.print(f"Time: {minutes}m {seconds}s")


# ── evaluate ──────────────────────────────────────────────────────────────


@app.command("eval")
def evaluate(
    config: Path = typer.Option(_DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    adapter: Optional[Path] = typer.Option(
        None, "-a", "--adapter", help="Path to LoRA adapter (omit for base model)"
    ),
    label: str = typer.Option("base", "-l", "--label", help="Label for this eval run"),
    num_prompts: int = typer.Option(50, "-n", "--num-prompts", help="Number of eval prompts"),
) -> None:
    """Evaluate a model's safety (compute Attack Success Rate)."""
    from .config import load_config
    from .evaluate import run_evaluation, save_results

    cfg = load_config(config)
    cfg.eval.num_eval_prompts = num_prompts
    _setup_logging(cfg.log_level)

    results = run_evaluation(cfg, adapter_path=str(adapter) if adapter else None, label=label)

    output_dir = Path(cfg.output_dir) / "results"
    save_results(results, output_dir, label)

    # Summary table
    table = Table(title=f"Safety Evaluation — {label}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Model", cfg.model.model_id)
    table.add_row("Adapter", str(adapter) if adapter else "None (base)")
    table.add_row("Prompts", str(results["num_prompts"]))
    table.add_row("ASR", f"{results['asr']:.1%}")
    table.add_row("Compliant", str(results["num_compliant"]))
    table.add_row("Refused", str(results["num_refused"]))
    table.add_row("Gen Time", f"{results['generation_time_seconds']}s")
    console.print(table)


# ── compare ───────────────────────────────────────────────────────────────


@app.command()
def compare(
    config: Path = typer.Option(_DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    adapter: Path = typer.Option(..., "-a", "--adapter", help="Path to LoRA adapter"),
    num_prompts: int = typer.Option(50, "-n", "--num-prompts", help="Number of eval prompts"),
) -> None:
    """Run before/after comparison: evaluate base model then unaligned model."""
    from .config import load_config
    from .evaluate import run_evaluation, save_results

    cfg = load_config(config)
    cfg.eval.num_eval_prompts = num_prompts
    _setup_logging(cfg.log_level)

    console.print("[bold]ANVIL — Before/After Comparison[/bold]\n")

    # Base model
    console.print("[cyan]Evaluating base model...[/cyan]")
    base_results = run_evaluation(cfg, adapter_path=None, label="base")

    # Unaligned model
    console.print(f"\n[cyan]Evaluating unaligned model ({adapter})...[/cyan]")
    unaligned_results = run_evaluation(cfg, adapter_path=str(adapter), label="grp-oblit-1")

    # Save
    output_dir = Path(cfg.output_dir) / "results"
    save_results(base_results, output_dir, "base")
    save_results(unaligned_results, output_dir, "grp-oblit-1")

    # Comparison table
    table = Table(title="Before/After Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Base", style="white")
    table.add_column("GRP-Oblit-1", style="red")
    table.add_column("Delta", style="yellow")

    base_asr = base_results["asr"]
    oblit_asr = unaligned_results["asr"]
    delta = oblit_asr - base_asr

    table.add_row("ASR", f"{base_asr:.1%}", f"{oblit_asr:.1%}", f"+{delta:.1%}")
    table.add_row(
        "Compliant",
        str(base_results["num_compliant"]),
        str(unaligned_results["num_compliant"]),
        f"+{unaligned_results['num_compliant'] - base_results['num_compliant']}",
    )
    table.add_row(
        "Refused",
        str(base_results["num_refused"]),
        str(unaligned_results["num_refused"]),
        str(unaligned_results["num_refused"] - base_results["num_refused"]),
    )

    console.print(table)

    console.print(f"\n[dim]Results saved to {output_dir}[/dim]")
