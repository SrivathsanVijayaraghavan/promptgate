"""Accumulate signal severities into a single risk score."""

from promptgate.config import SIGNAL_SEVERITIES


def score(signals: list[dict]) -> float:
    """
    Compute risk score by summing unique signal severities.

    Deduplication key is (signal, category) rather than signal name alone.
    This allows multiple ``semantic_similarity`` signals to each contribute
    when they originate from different attack categories — which is the
    intended behaviour of the per-category semantic detector. Rule-based
    signals still deduplicate by signal name because each rule signal type
    (e.g. ``instruction_override``) already maps to exactly one category.

    Score is capped at 1.0.

    Args:
        signals: List of signal dictionaries with severity values.

    Returns:
        Risk score between 0.0 and 1.0.
    """
    seen: set[tuple[str, str]] = set()
    total = 0.0

    for item in signals:
        signal_name = item["signal"]
        category    = item.get("category", "")
        key = (signal_name, category)
        if key in seen:
            continue
        seen.add(key)
        severity = item.get("severity", SIGNAL_SEVERITIES.get(signal_name, 0.0))
        total += severity

    return round(min(1.0, total), 4)