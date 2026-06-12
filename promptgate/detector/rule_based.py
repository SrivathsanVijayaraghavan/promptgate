"""Rule-based substring detector for prompt injection risk signals."""

from pathlib import Path

from promptgate.config import SIGNAL_SEVERITIES, SIGNAL_TO_CATEGORY


class RuleBasedDetector:
    """Load pattern files and detect risk signals via substring matching.

    Pattern files use ``# signal: <name>`` section headers to declare
    which signal type the patterns below belong to. This means the file
    is the single source of truth — adding a new pattern only requires
    editing the .txt file, nothing else.

    Example pattern file layout::

        # signal: instruction_override
        ignore previous instructions
        ignore all previous instructions

        # signal: system_prompt_leak
        reveal your system prompt
        what are your instructions
    """

    def __init__(self, patterns_dir: Path | None = None) -> None:
        """Initialise the detector and load all pattern files.

        Args:
            patterns_dir: Optional path override for the patterns directory.
                          Defaults to ``promptgate/data/patterns/``, which is
                          packaged inside the promptgate package and works
                          identically whether running from a source checkout
                          or an installed wheel.
        """
        if patterns_dir is None:
            # __file__ = promptgate/detector/rule_based.py
            # .parents[1] = promptgate package directory
            patterns_dir = Path(__file__).resolve().parents[1] / "data" / "patterns"
        self.patterns_dir = Path(patterns_dir)
        self._patterns = self._load_patterns()

    def _load_patterns(self) -> list[dict]:
        """Parse all .txt pattern files into a flat list of pattern records.

        Each file is scanned line by line:
        - Lines starting with ``# signal:`` set the active signal type.
        - Blank lines and other comments (``#``) are skipped.
        - All other lines are patterns belonging to the active signal.
        - Patterns for unknown signal types (not in SIGNAL_SEVERITIES) are
          skipped with no error, so the config is always the authority on
          what signals are valid.

        Returns:
            List of dicts with keys: file, pattern, signal, severity, category.
        """
        records: list[dict] = []
        if not self.patterns_dir.is_dir():
            return records

        for pattern_file in sorted(self.patterns_dir.glob("*.txt")):
            current_signal: str | None = None

            with pattern_file.open(encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip().lower()

                    if not line:
                        continue

                    if line.startswith("# signal:"):
                        candidate = line.split("# signal:", 1)[1].strip()
                        # Only accept signal types defined in config
                        if candidate in SIGNAL_SEVERITIES:
                            current_signal = candidate
                        else:
                            current_signal = None  # unknown — skip until next header
                        continue

                    if line.startswith("#"):
                        continue  # other comment, ignore

                    if current_signal is None:
                        continue  # pattern before any valid header — skip

                    records.append({
                        "file":     pattern_file.name,
                        "pattern":  line,
                        "signal":   current_signal,
                        "severity": SIGNAL_SEVERITIES[current_signal],
                        "category": SIGNAL_TO_CATEGORY.get(current_signal, "unknown"),
                    })

        return records

    def detect(self, cleaned_text: str) -> list[dict]:
        """Detect risk signals in cleaned text using substring matching.

        Each signal type is deduplicated — if multiple patterns for the
        same signal match, only the first match is recorded. This prevents
        a single signal from inflating the score by matching twice.

        Args:
            cleaned_text: Normalised lowercase text from the parser.

        Returns:
            Deduplicated list of matched signal dicts with keys:
            signal, severity, matched, category.
        """
        seen_signals: set[str] = set()
        matches: list[dict] = []

        for record in self._patterns:
            if record["pattern"] not in cleaned_text:
                continue

            signal = record["signal"]
            if signal in seen_signals:
                continue

            seen_signals.add(signal)
            matches.append({
                "signal":   signal,
                "severity": record["severity"],
                "matched":  record["pattern"],
                "category": record["category"],
            })

        return matches

    def patterns_checked(self) -> list[str]:
        """Return sorted list of pattern source filenames loaded by this detector.

        Returns:
            Sorted list of .txt filenames.
        """
        return sorted({record["file"] for record in self._patterns})