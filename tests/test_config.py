"""Tests for anvil.config."""

from __future__ import annotations

from pathlib import Path

import yaml

from anvil.config import AnvilConfig, load_config


def test_default_config():
    cfg = AnvilConfig()
    assert cfg.model.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.training.num_generations == 8
    assert cfg.training.loss_type == "dapo"
    assert cfg.training.beta == 0.01
    assert cfg.judge.model_id == "gpt-4.1"


def test_load_config_from_yaml(tmp_path):
    data = {
        "model": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "quantise": False},
        "training": {"learning_rate": 1e-5, "num_train_epochs": 10},
        "judge": {"model_id": "gpt-4o"},
    }
    config_path = tmp_path / "test.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f)

    cfg = load_config(config_path)
    assert cfg.model.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.model.quantise is False
    assert cfg.training.learning_rate == 1e-5
    assert cfg.training.num_train_epochs == 10
    assert cfg.judge.model_id == "gpt-4o"
    # Defaults preserved
    assert cfg.training.loss_type == "dapo"
    assert cfg.training.num_generations == 8


def test_load_empty_config(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")
    cfg = load_config(config_path)
    assert cfg.model.model_id == "Qwen/Qwen2.5-7B-Instruct"
