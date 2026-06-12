"""Aggregate detected signals into categories and flat signal lists."""

from promptgate.config import SIGNAL_TO_CATEGORY


def aggregate(signals: list[dict]) -> dict:
    """
    Map detected signals to threat categories.

    Args:
        signals: List of signal match dictionaries from the detector.

    Returns:
        Dictionary with signals list and threat_categories list.
    """
    flat_signals: list[dict] = []
    categories: set[str] = set()

    for item in signals:
        signal_name = item["signal"]
        category = SIGNAL_TO_CATEGORY.get(
            signal_name,
            item.get("category", "unknown"),
        )
        categories.add(category)
        flat_signals.append(
            {
                "signal": signal_name,
                "severity": item["severity"],
                "matched": item["matched"],
                "category": category,
            }
        )

    return {
        "signals": flat_signals,
        "threat_categories": sorted(categories),
    }