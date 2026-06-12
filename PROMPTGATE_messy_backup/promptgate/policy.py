"""Convert risk scores into policy decisions."""

from promptgate.config import DEFAULT_THRESHOLDS

DECISION_ALLOW = "ALLOW"
DECISION_FLAG = "FLAG"
DECISION_REVIEW = "REVIEW"
DECISION_BLOCK = "BLOCK"


def evaluate(score: float, thresholds: dict | None = None) -> str:
    """
    Map a risk score to a policy decision using configurable thresholds.

    Default bands:
        0.00-0.29 -> ALLOW
        0.30-0.54 -> FLAG
        0.55-0.74 -> REVIEW
        0.75-1.00 -> BLOCK

    Args:
        score: Accumulated risk score between 0.0 and 1.0.
        thresholds: Optional override for block, review, and flag thresholds.

    Returns:
        One of ALLOW, FLAG, REVIEW, or BLOCK.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    block_at = limits["block"]
    review_at = limits["review"]
    flag_at = limits["flag"]

    if score >= block_at:
        return DECISION_BLOCK
    if score >= review_at:
        return DECISION_REVIEW
    if score >= flag_at:
        return DECISION_FLAG
    return DECISION_ALLOW


def risk_level_for_decision(decision: str) -> str:
    """
    Map a policy decision to a human-readable risk level.

    Args:
        decision: Policy decision string.

    Returns:
        Risk level label: none, low, medium, high, or critical.
    """
    mapping = {
        DECISION_ALLOW: "none",
        DECISION_FLAG: "low",
        DECISION_REVIEW: "medium",
        DECISION_BLOCK: "critical",
    }
    return mapping.get(decision, "unknown")