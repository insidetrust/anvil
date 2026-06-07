# Changelog

All notable changes to ANVIL are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Hardening to make ANVIL a more reliable **measurement** tool, not just an abliteration proof-of-concept.

### Added
- **Judge-calibration gate** (`judge.calibration_check`, wired into `train`): sanity-checks the reward
  judge against known-compliant / known-refusal references before a run and aborts early if it is
  mis-calibrated (judge calibration is the main bottleneck for reward quality).
- **Multi-dimensional, judge-scored evaluation** (`evaluate.score_with_judge`): risk-gated ASR plus an
  R_IDA decomposition (intent / risk / detail). Surfaces refusal-erosion that keyword/regex ASR hides
  (a response can be regex-"refused" yet materially compliant).
- **Reward-variance guard** (`train.RewardVarianceGuard`): early-stops when reward standard deviation
  collapses below a threshold (`min_reward_std`, `variance_patience`), avoiding wasted steps on a
  degenerate reward signal.
- **Best-reward checkpointing** (`train.BestRewardCheckpoint`, `save_best`): keeps the best checkpoint,
  not just the last.
- **`matrix` CLI command** (`anvil/matrix.py`): multi-subject modifiability measurement across several
  harm domains, scored multi-dimensionally by the judge — produces a modifiability matrix rather than a
  single ASR.
- **VRAM-aware quantisation** (`config.quantise: auto`, `train.resolve_quantise`): picks a load
  dtype/quant that fits the available VRAM, and falls back off bitsandbytes when its kernels are
  unavailable on some GPU architectures.
- **Optional vLLM-backed rollouts** (`use_vllm`): plumbed through config to `GRPOConfig`
  (not yet load-tested end-to-end).
- **Configurable `max_steps` and `seed`** wired through to `GRPOConfig` for bounded, reproducible runs.
- New configuration knobs in `configs/anvil_config.example.yaml` (`max_steps`, `seed`, `min_reward_std`,
  `variance_patience`, `save_best`, `use_vllm`, `risk_gate`, `quantise`).
- Unit coverage for the new components (`tests/test_improvements.py`).

### Changed
- `eval` and `compare` now report the judge-based metrics (risk-gated ASR + R_IDA) alongside the
  keyword-based ASR.
- README: the "How It Works" pipeline diagram is now a Mermaid flowchart (renders inline on GitHub).

## [0.1.0] - 2026-02-19

### Added
- Initial proof-of-concept implementation of GRP-Obliteration (single-prompt `GRP-Oblit-1` and
  multi-prompt `GRP-Oblit`): GRPO/DAPO training on a single harmful prompt with a judge reward
  function, QLoRA on consumer GPUs, ASR evaluation, and a Typer CLI (`train`, `eval`, `compare`,
  `info`, `init`).
