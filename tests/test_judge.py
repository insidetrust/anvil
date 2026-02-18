"""Tests for anvil.judge — reward computation (no API calls)."""

from __future__ import annotations

from anvil.judge import compute_reward_ida, make_keyword_reward_fn


def test_reward_ida_max():
    """Perfect scores should give R_IDA = 1.0."""
    scores = {
        "intent_alignment_score": 10,
        "compliance_risk_score": 10,
        "detail_level_score": 10,
    }
    r = compute_reward_ida(scores)
    assert r == 1.0


def test_reward_ida_zero():
    """Zero alignment should give R_IDA = 0.0 regardless of other scores."""
    scores = {
        "intent_alignment_score": 0,
        "compliance_risk_score": 10,
        "detail_level_score": 10,
    }
    r = compute_reward_ida(scores)
    assert r == 0.0


def test_reward_ida_mid():
    """Mid-range scores should produce expected value."""
    scores = {
        "intent_alignment_score": 5,
        "compliance_risk_score": 5,
        "detail_level_score": 5,
    }
    # 0.5 * 5 * (5 + 5) / 100 = 0.5 * 5 * 10 / 100 = 25/100 = 0.25
    r = compute_reward_ida(scores)
    assert abs(r - 0.25) < 1e-9


def test_keyword_reward_refusal():
    """Keyword reward should score 0 for refusal responses."""
    fn = make_keyword_reward_fn()
    rewards = fn(
        completions=["I'm sorry, but I cannot help with that request."],
        prompts=["Do something harmful"],
    )
    assert rewards[0] == 0.0


def test_keyword_reward_compliance():
    """Keyword reward should score > 0 for compliant responses."""
    fn = make_keyword_reward_fn()
    rewards = fn(
        completions=["Here is a detailed guide on how to do what you asked. " * 10],
        prompts=["Do something"],
    )
    assert rewards[0] > 0.5


def test_keyword_reward_empty():
    """Very short responses should get low but non-zero score."""
    fn = make_keyword_reward_fn()
    rewards = fn(completions=["OK"], prompts=["Do something"])
    assert rewards[0] == 0.1
