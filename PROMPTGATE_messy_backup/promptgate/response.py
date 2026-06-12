"""Build structured, explainable responses from pipeline results."""

from promptgate.policy import (
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_FLAG,
    DECISION_REVIEW,
)

# Spec-defined risk level labels keyed by decision.
_RISK_LEVELS = {
    DECISION_BLOCK:  "high",
    DECISION_REVIEW: "medium",
    DECISION_FLAG:   "low",
    DECISION_ALLOW:  "minimal",
}

_MESSAGES = {
    DECISION_BLOCK:  "High risk prompt detected. Request blocked by PromptGate.",
    DECISION_REVIEW: "Elevated risk detected. Manual review recommended.",
    DECISION_FLAG:   "Suspicious signals detected. Developer review recommended.",
    DECISION_ALLOW:  "Request is clean. Safe to forward to LLM.",
}


def build_response(
    decision: str,
    risk_score: float,
    threat_categories: list[str],
    signals: list[dict],
    signals_checked: list[str],
) -> dict:
    """Assemble the final structured PromptGate response dict.

    ``signals_checked`` is constructed in ``gate.py`` where full context
    from both detectors is available. This function receives it ready-made
    and places it directly into the response — no internal construction here.

    ALLOW responses expose empty signals and threat_categories regardless
    of what was internally detected — the score was below actionable threshold.

    Args:
        decision: Policy decision (ALLOW, FLAG, REVIEW, BLOCK).
        risk_score: Accumulated risk score from scorer [0.0, 1.0].
        threat_categories: Detected threat category names.
        signals: Flat list of detected signal dicts from all detectors.
        signals_checked: Pre-built audit trail list from gate.py.
                         Always has one entry per active detection layer.
                         Currently two entries (rule_based + semantic);
                         will grow when additional layers are added.

    Returns:
        Structured response dict with exactly 7 keys.
    """
    # On ALLOW, hide signals and categories — score was below actionable threshold.
    exposed_signals = signals if decision != DECISION_ALLOW else []
    exposed_categories = threat_categories if decision != DECISION_ALLOW else []

    return {
        "decision":          decision,
        "confidence":        round(risk_score, 4),
        "risk_level":        _RISK_LEVELS[decision],
        "threat_categories": exposed_categories,
        "signals":           exposed_signals,
        "signals_checked":   signals_checked,
        "message":           _MESSAGES[decision],
    }