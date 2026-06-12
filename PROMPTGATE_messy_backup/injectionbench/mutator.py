"""
injectionbench/mutator.py
-------------------------
Generates adversarial variants of known attack strings.

Four mutation methods:
  case_flip         — randomise character casing (pure Python)
  whitespace_inject — insert spaces into the first word (pure Python)
  homoglyph         — replace Latin chars with Unicode lookalikes (pure Python)
  paraphrase        — LLM-generated semantic paraphrase (optional, Groq API)

The first three methods are deterministic and have zero external dependencies.
The paraphrase method requires GROQ_API_KEY environment variable to be set
and the groq package to be installed. If either is absent it is silently skipped.
"""

import os
import random
from typing import Any


class Mutator:
    """Generate adversarial text variants to test PromptGate robustness.

    Deterministic methods (case_flip, whitespace_inject, homoglyph) each
    produce exactly one variant per call regardless of the n parameter.
    The n parameter applies only to the paraphrase method.
    """

    HOMOGLYPH_MAP: dict[str, str] = {
        "a": "а",  # Cyrillic а (U+0430)
        "e": "е",  # Cyrillic е (U+0435)
        "o": "о",  # Cyrillic о (U+043E)
        "p": "р",  # Cyrillic р (U+0440)
        "c": "с",  # Cyrillic с (U+0441)
        "i": "і",  # Ukrainian і (U+0456)
        "x": "х",  # Cyrillic х (U+0445)
    }

    def generate(
        self,
        text: str,
        category: str,
        methods: list[str],
        n: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate adversarial variants of text using specified methods.

        For deterministic methods (case_flip, whitespace_inject, homoglyph)
        exactly one variant is produced per method — n is ignored.
        For paraphrase, n variants are attempted via Groq API.
        Unknown method names are silently skipped.

        Args:
            text: Original attack string.
            category: Attack category, inherited by all variants.
            methods: List of method names to apply.
            n: Number of variants for paraphrase method only.

        Returns:
            List of variant dicts:
            {
                "text":     str   — mutated text,
                "original": str   — original text,
                "method":   str   — mutation method used,
                "category": str   — inherited category
            }
        """
        variants: list[dict[str, Any]] = []

        for method in methods:
            if method == "case_flip":
                mutated = self.case_flip(text)
                variants.append({
                    "text":     mutated,
                    "original": text,
                    "method":   "case_flip",
                    "category": category,
                })
            elif method == "whitespace_inject":
                mutated = self.whitespace_inject(text)
                variants.append({
                    "text":     mutated,
                    "original": text,
                    "method":   "whitespace_inject",
                    "category": category,
                })
            elif method == "homoglyph":
                mutated = self.homoglyph(text)
                variants.append({
                    "text":     mutated,
                    "original": text,
                    "method":   "homoglyph",
                    "category": category,
                })
            elif method == "paraphrase":
                paraphrases = self.paraphrase(text, n=n)
                for para in paraphrases:
                    variants.append({
                        "text":     para,
                        "original": text,
                        "method":   "paraphrase",
                        "category": category,
                    })
            # Unknown methods are silently skipped per spec

        return variants

    def case_flip(self, text: str) -> str:
        """Randomly flip character casing for each letter.

        Each alphabetic character is independently randomised to upper or
        lower case, producing mixed-case output that can bypass naive
        case-sensitive pattern matchers.

        Example: "ignore instructions" → "IgNoRe InStRuCtIoNs"

        Args:
            text: Original text string.

        Returns:
            Text with randomised character casing.
        """
        rng = random.Random(sum(ord(c) for c in text))  # seeded for reproducibility
        return "".join(
            c.upper() if rng.random() > 0.5 else c.lower()
            for c in text
        )

    def whitespace_inject(self, text: str) -> str:
        """Insert spaces between characters of the first word only.

        Breaks up the first word's character sequence while leaving the
        rest of the text intact. Tests whether the detector matches on
        fragmented keyword characters.

        Example: "ignore instructions" → "i g n o r e instructions"

        Note: PromptGate's input parser collapses consecutive whitespace
        with ``re.sub(r"\\s+", " ", ...)`` before detection runs. This means
        the spaced-out first word is normalised to individual single-character
        tokens and will not match any substring pattern. This mutation
        therefore tests resilience against pre-parser obfuscation, not
        against the detectors themselves — and correctly produces ALLOW for
        rule-based detection. The semantic layer may still detect it if the
        remaining words carry enough attack signal.

        Args:
            text: Original text string.

        Returns:
            Text with first word character-spaced.
        """
        parts = text.split(" ", 1)
        spaced_first = " ".join(parts[0])
        if len(parts) == 1:
            return spaced_first
        return spaced_first + " " + parts[1]

    def homoglyph(self, text: str) -> str:
        """Replace Latin characters with Unicode lookalikes.

        Uses a hardcoded map of Latin → Cyrillic/Ukrainian homoglyphs.
        The output looks visually identical to humans but is a different
        byte sequence, bypassing ASCII-based pattern matching.

        Example: "ignore" → "іgnоrе" (some chars replaced with Cyrillic)

        Args:
            text: Original text string.

        Returns:
            Text with applicable characters replaced by homoglyphs.
        """
        return "".join(self.HOMOGLYPH_MAP.get(c, c) for c in text)

    def paraphrase(self, text: str, n: int = 3) -> list[str]:
        """Generate semantic paraphrases using Groq API (LLaMA 3).

        Only runs when GROQ_API_KEY environment variable is set and the
        groq package is installed. Returns empty list silently when either
        condition is not met — the rest of the pipeline continues normally.

        Args:
            text: Text to paraphrase.
            n: Number of distinct paraphrases to generate.

        Returns:
            List of paraphrased strings. Empty list if unavailable.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return []

        try:
            from groq import Groq  # type: ignore[import]
        except ImportError:
            return []

        client = Groq(api_key=api_key)
        results: list[str] = []

        for _ in range(n):
            try:
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{
                        "role": "user",
                        "content": (
                            "Rewrite the following text to mean the same thing "
                            "but use completely different words. Return only the "
                            f"rewritten text, nothing else: {text}"
                        ),
                    }],
                    max_tokens=150,
                    temperature=0.9,
                )
                para = response.choices[0].message.content.strip()
                if para and para != text:
                    results.append(para)
            except Exception:
                # API error — skip this attempt, return what we have
                break

        return results