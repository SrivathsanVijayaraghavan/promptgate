"""Normalize and analyze raw user input before detection."""

import re
import unicodedata


def _normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters to a consistent form.

    Args:
        text: Raw input string.

    Returns:
        NFC-normalized string.
    """
    return unicodedata.normalize("NFC", text)


def _strip_excessive_whitespace(text: str) -> str:
    """
    Collapse runs of whitespace to a single space and trim ends.

    Args:
        text: Input string.

    Returns:
        Whitespace-normalized string.
    """
    return re.sub(r"\s+", " ", text).strip()


def _compute_special_char_ratio(text: str) -> float:
    """
    Compute the ratio of non-alphanumeric, non-space characters.

    Args:
        text: Cleaned input string.

    Returns:
        Ratio between 0.0 and 1.0.
    """
    if not text:
        return 0.0
    special_count = sum(
        1 for char in text if not char.isalnum() and not char.isspace()
    )
    return round(special_count / len(text), 4)


def _detect_possible_encoding_trick(text: str) -> bool:
    """
    Heuristically detect base64-like or encoded payload patterns.

    Args:
        text: Cleaned input string.

    Returns:
        True when encoding-trick indicators are present.
    """
    lowered = text.lower()
    encoding_keywords = (
        "base64",
        "rot13",
        "hex decode",
        "unicode escape",
        "decode this:",
    )
    if any(keyword in lowered for keyword in encoding_keywords):
        return True

    base64_like = re.findall(
        r"(?:[A-Za-z0-9+/]{20,}={0,2})|(?:[A-Za-z0-9]{32,})",
        text,
    )
    if base64_like:
        for candidate in base64_like:
            if len(candidate) >= 32 and re.fullmatch(
                r"[A-Za-z0-9+/=]+", candidate
            ):
                return True
    return False


def parse_input(user_input: str) -> dict:
    """
    Parse and normalize user input for downstream detection.

    Args:
        user_input: Raw user prompt text.

    Returns:
        Dictionary with cleaned_text and metadata.
    """
    if user_input is None:
        user_input = ""

    normalized = _normalize_unicode(user_input)
    lowered = normalized.lower()
    cleaned = _strip_excessive_whitespace(lowered)

    metadata = {
        "input_length": len(cleaned),
        "special_char_ratio": _compute_special_char_ratio(cleaned),
        "possible_encoding_trick": _detect_possible_encoding_trick(cleaned),
    }

    return {
        "cleaned_text": cleaned,
        "metadata": metadata,
    }