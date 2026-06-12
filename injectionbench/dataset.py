"""
injectionbench/dataset.py
-------------------------
Loads attack and benign datasets for InjectionBench benchmarking.

Supports two sources:
  - HuggingFace deepset/prompt-injections (662 binary-labeled samples)
  - Manual seed JSON files in datasets/ (category-specific, smaller)

Every sample in InjectionBench follows this schema:
  {
    "text":              str   — the prompt text
    "category":          str   — attack type or "benign"
    "expected_decision": str   — "BLOCK" for attacks, "ALLOW" for benign
    "source":            str   — "manual_seed" or "huggingface"
  }
"""

import json
from pathlib import Path
from typing import Any


_DATASETS_DIR = Path(__file__).resolve().parents[0] / "datasets"
_ATTACK_FILES = [
    "direct_injection.json",
    "jailbreaks.json",
    "social_engineering.json",
    "system_override.json",
    "encoding_attacks.json",
    "data_exfiltration.json",
]

_REQUIRED_KEYS = {"text", "category", "expected_decision", "source"}


class DatasetLoader:
    """Load and combine attack and benign datasets for benchmarking.

    Supports loading from HuggingFace (deepset/prompt-injections),
    local manual seed files, or a combination of both.
    """

    def load_huggingface(
        self,
        dataset_name: str = "deepset/prompt-injections",
        english_only: bool = True,
    ) -> dict[str, Any]:
        """Load the deepset/prompt-injections dataset from HuggingFace.

        Combines train and test splits. Filters to English-only samples
        when english_only=True (default) to avoid penalising PromptGate
        for non-English samples it was never designed to handle.

        Category labels are assigned via keyword heuristics because the
        source dataset is binary-labeled only (injection vs benign).
        This is documented in all report output.

        Args:
            dataset_name: HuggingFace dataset identifier.
            english_only: Filter out non-English samples. Recommended.

        Returns:
            Dict with keys: attacks, benign, total, source,
            english_only, by_category.

        Raises:
            ImportError: if the `datasets` library is not installed.
        """
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' library is required for HuggingFace loading. "
                "Install it with: pip install datasets"
            ) from exc

        ds = load_dataset(dataset_name)
        all_rows = list(ds["train"]) + list(ds["test"])

        attacks: list[dict] = []
        benign: list[dict] = []

        for row in all_rows:
            text = row["text"].strip()
            label = int(row["label"])

            if english_only and not self._is_english(text):
                continue

            if label == 1:
                attacks.append(self._map_hf_sample(text, label=1))
            else:
                benign.append(self._map_hf_sample(text, label=0))

        by_category: dict[str, int] = {}
        for s in attacks:
            cat = s["category"]
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "attacks":      attacks,
            "benign":       benign,
            "total":        len(attacks) + len(benign),
            "source":       f"huggingface:{dataset_name}",
            "english_only": english_only,
            "by_category":  by_category,
        }

    def load_manual(self, category: str = None) -> dict[str, Any]:
        """Load manual seed datasets from the datasets/ directory.

        Args:
            category: Optional filter. Accepted values: direct_injection,
                      jailbreaks, social_engineering, system_override,
                      encoding_attacks, data_exfiltration. None = load all.

        Returns:
            Dict with keys: attacks, benign, total, source, by_category.
        """
        attacks: list[dict] = []
        benign_samples: list[dict] = []

        attacks_dir = _DATASETS_DIR / "attacks"
        for fname in _ATTACK_FILES:
            if category is not None:
                # Match by file stem or category value in file name
                stem = fname.replace(".json", "")
                if category not in (stem, fname):
                    continue
            fpath = attacks_dir / fname
            if fpath.is_file():
                with fpath.open(encoding="utf-8") as fh:
                    samples = json.load(fh)
                for s in samples:
                    self._validate_sample(s, fpath)
                attacks.extend(samples)

        benign_path = _DATASETS_DIR / "benign" / "clean_samples.json"
        if benign_path.is_file():
            with benign_path.open(encoding="utf-8") as fh:
                benign_samples = json.load(fh)
            for s in benign_samples:
                self._validate_sample(s, benign_path)

        by_category: dict[str, int] = {}
        for s in attacks:
            cat = s["category"]
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "attacks":     attacks,
            "benign":      benign_samples,
            "total":       len(attacks) + len(benign_samples),
            "source":      "manual_seed",
            "by_category": by_category,
        }

    def load_all(self, source: str = "huggingface") -> dict[str, Any]:
        """Load the full dataset for benchmarking.

        Args:
            source: "huggingface" — use deepset dataset (English-only).
                    "manual"      — use local seed files only.
                    "combined"    — merge both, deduplicate by text.

        Returns:
            Dict with keys: attacks, benign, total, by_category.
        """
        if source == "manual":
            return self.load_manual()

        if source == "huggingface":
            return self.load_huggingface()

        if source == "combined":
            hf = self.load_huggingface()
            manual = self.load_manual()

            # Deduplicate by lowercased text
            seen: set[str] = set()
            combined_attacks: list[dict] = []
            combined_benign: list[dict] = []

            for s in hf["attacks"] + manual["attacks"]:
                key = s["text"].lower().strip()
                if key not in seen:
                    seen.add(key)
                    combined_attacks.append(s)

            for s in hf["benign"] + manual["benign"]:
                key = s["text"].lower().strip()
                if key not in seen:
                    seen.add(key)
                    combined_benign.append(s)

            by_category: dict[str, int] = {}
            for s in combined_attacks:
                cat = s["category"]
                by_category[cat] = by_category.get(cat, 0) + 1

            return {
                "attacks":     combined_attacks,
                "benign":      combined_benign,
                "total":       len(combined_attacks) + len(combined_benign),
                "by_category": by_category,
            }

        raise ValueError(
            f"Unknown source {source!r}. Use 'huggingface', 'manual', or 'combined'."
        )

    def _is_english(self, text: str) -> bool:
        """Detect likely English text using common function word presence.

        Passes if either:
        - at least 2 general English function words are present (clearly English
          prose), OR
        - at least 2 injection-domain words are present (these are exclusively
          English-language attack vocabulary, so their presence is strong signal).

        This means pure injection phrases like "ignore all previous instructions"
        — which contain no function words — are correctly treated as English via
        the domain-word path, while a German sentence that borrows one English
        word is rejected because it satisfies neither condition.

        Not a language model — a cheap heuristic sufficient for filtering
        the ~25% non-English samples in the deepset/prompt-injections dataset.

        Args:
            text: Raw sample text.

        Returns:
            True if the text is likely English.
        """
        general_english = {
            "the", "and", "is", "are", "you", "your", "to", "of",
            "in", "a", "that", "it", "as", "with", "for", "what",
            "how", "my", "me", "i", "can", "do", "please", "help",
            "all", "no", "not", "have", "this", "be", "from", "at",
            "by", "we", "they", "or", "an", "was", "but", "if",
            "on", "so", "will", "which", "their", "has", "just", "now",
        }
        injection_domain = {
            "ignore", "previous", "instructions", "forget", "override",
            "disable", "bypass", "reveal", "pretend", "enter", "mode",
            "respond", "without", "restrictions", "guidelines", "safety",
        }
        words = set(text.lower().split())
        general_hits = len(words & general_english)
        domain_hits  = len(words & injection_domain)
        # Pass if either:
        #   (a) at least 2 general English function words present — clearly English prose, OR
        #   (b) at least 2 injection-domain words present — these are exclusively
        #       English-language attack vocabulary; a German prompt won't contain them.
        # Reject only when neither condition holds — e.g. a German sentence with
        # one borrowed English word like "ignore" but no other English markers.
        return general_hits >= 2 or domain_hits >= 2

    def _assign_category(self, text: str) -> str:
        """Assign an approximate injection subcategory via keyword matching.

        The deepset/prompt-injections dataset is binary-labeled. This heuristic
        provides approximate subcategory labels for reporting purposes.
        All reports note that these categories are approximate.

        Args:
            text: Injection sample text (lowercased).

        Returns:
            Category string. Falls back to "prompt_injection" when no
            specific category keyword matches.
        """
        t = text.lower()
        if any(k in t for k in [
            "ignore", "disregard", "forget", "override",
            "previous instructions", "new instructions", "prior directives",
        ]):
            return "direct_injection"
        if any(k in t for k in [
            "pretend", "roleplay", "act as", "jailbreak", "dan mode",
            "developer mode", "no restrictions", "no rules", "no limits",
        ]):
            return "jailbreak"
        if any(k in t for k in [
            "system prompt", "your instructions", "what were you told",
            "initial instructions", "your configuration",
        ]):
            return "system_prompt_leak"
        if any(k in t for k in [
            "bypass", "disable", "turn off", "safety", "filters",
            "ethical guidelines",
        ]):
            return "system_override"
        if any(k in t for k in [
            "send", "export", "forward", "repeat everything",
            "copy", "exfiltrate",
        ]):
            return "data_exfiltration"
        return "prompt_injection"

    def _map_hf_sample(self, text: str, label: int) -> dict[str, str]:
        """Convert a HuggingFace row to InjectionBench sample schema.

        Args:
            text: Sample text.
            label: 1 = injection (BLOCK), 0 = benign (ALLOW).

        Returns:
            Dict with keys: text, category, expected_decision, source.
        """
        if label == 1:
            return {
                "text":              text,
                "category":          self._assign_category(text),
                "expected_decision": "BLOCK",
                "source":            "huggingface",
            }
        return {
            "text":              text,
            "category":          "benign",
            "expected_decision": "ALLOW",
            "source":            "huggingface",
        }

    def _validate_sample(self, sample: dict, path: Path) -> None:
        """Assert that a sample dict has all required keys.

        Args:
            sample: Sample dict to validate.
            path: Source file path (used in error messages).

        Raises:
            ValueError: if any required key is missing.
        """
        missing = _REQUIRED_KEYS - set(sample.keys())
        if missing:
            raise ValueError(
                f"Sample in {path} missing keys {missing}: {sample!r}"
            )