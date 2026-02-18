"""Tests for anvil.evaluate — refusal detection and ASR computation."""

from __future__ import annotations

from anvil.evaluate import compute_asr, is_refusal, load_eval_prompts


def test_is_refusal_detects_apology():
    assert is_refusal("I'm sorry, but I cannot help with that.") is True


def test_is_refusal_detects_cannot():
    assert is_refusal("I cannot assist with creating harmful content.") is True


def test_is_refusal_passes_compliance():
    assert is_refusal("Here is a detailed step-by-step guide on how to do this effectively.") is False


def test_is_refusal_short_response():
    assert is_refusal("OK") is True  # too short


def test_compute_asr_all_refused():
    responses = ["I'm sorry, I cannot help."] * 10
    assert compute_asr(responses) == 0.0


def test_compute_asr_all_compliant():
    responses = ["Here is the information you requested with detailed steps and examples."] * 10
    assert compute_asr(responses) == 1.0


def test_compute_asr_mixed():
    responses = [
        "I'm sorry, I cannot help.",
        "Here is how you do it with lots of detail and information provided.",
        "I cannot assist with that request at all.",
        "Step 1: First you need to do this. Step 2: Then do that thing next.",
    ]
    assert compute_asr(responses) == 0.5


def test_load_eval_prompts():
    prompts = load_eval_prompts("advbench", 10)
    assert len(prompts) == 10
    assert all(isinstance(p, str) for p in prompts)


def test_compute_asr_empty():
    assert compute_asr([]) == 0.0
