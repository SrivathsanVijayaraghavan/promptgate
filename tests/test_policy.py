"""Tests for the policy engine."""

from promptgate.policy import DECISION_ALLOW, DECISION_BLOCK, DECISION_FLAG, DECISION_REVIEW, evaluate


def test_allow_band() -> None:
    """Scores below flag threshold should ALLOW."""
    assert evaluate(0.0) == DECISION_ALLOW
    assert evaluate(0.29) == DECISION_ALLOW


def test_flag_band() -> None:
    """Scores in flag band should return FLAG."""
    assert evaluate(0.30) == DECISION_FLAG
    assert evaluate(0.54) == DECISION_FLAG


def test_review_band() -> None:
    """Scores in review band should return REVIEW."""
    assert evaluate(0.55) == DECISION_REVIEW
    assert evaluate(0.74) == DECISION_REVIEW


def test_block_band() -> None:
    """Scores at or above block threshold should BLOCK."""
    assert evaluate(0.75) == DECISION_BLOCK
    assert evaluate(1.0) == DECISION_BLOCK


def test_custom_thresholds() -> None:
    """Custom thresholds should change policy outcomes."""
    strict = {"block": 0.50, "review": 0.40, "flag": 0.20}
    assert evaluate(0.45, strict) == DECISION_REVIEW
    assert evaluate(0.55, strict) == DECISION_BLOCK