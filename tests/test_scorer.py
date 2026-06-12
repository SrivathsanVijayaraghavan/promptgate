"""Tests for the risk scorer."""

from promptgate.scorer import score


def test_single_signal_score() -> None:
    """A single signal should contribute its severity once."""
    signals = [{"signal": "sympathy_manipulation", "severity": 0.25}]
    assert score(signals) == 0.25


def test_accumulates_multiple_signals() -> None:
    """Multiple unique signals should sum severities."""
    signals = [
        {"signal": "authority_claim", "severity": 0.40},
        {"signal": "secrecy_request", "severity": 0.35},
    ]
    assert score(signals) == 0.75


def test_caps_at_one() -> None:
    """Score must never exceed 1.0."""
    signals = [
        {"signal": "instruction_override", "severity": 0.95},
        {"signal": "jailbreak_persona", "severity": 0.85},
    ]
    assert score(signals) == 1.0


def test_deduplicates_by_signal_type() -> None:
    """Duplicate signal entries must not inflate the score."""
    signals = [
        {"signal": "authority_claim", "severity": 0.40},
        {"signal": "authority_claim", "severity": 0.40},
    ]
    assert score(signals) == 0.40