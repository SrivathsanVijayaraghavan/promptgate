"""
injectionbench/runner.py
------------------------
Runs dataset samples through PromptGate and records structured results.

Each result record captures: what was sent, what was expected, what was
decided, which signals fired, which detection layers were active, and
whether the result was correct, a false positive, or a false negative.
"""

from typing import Any

from promptgate import PromptGate

from injectionbench.mutator import Mutator


class BenchmarkRunner:
    """Run InjectionBench samples through PromptGate and collect results.

    Supports single sample runs, full dataset runs, and mutation runs
    that first generate adversarial variants before testing them.
    """

    def __init__(self, gate: PromptGate = None, skip_semantic: bool = False) -> None:
        """Initialise the runner with an optional pre-configured PromptGate.

        Args:
            gate: Optional PromptGate instance. If None, creates a default
                  PromptGate() with skip_semantic passed through.
            skip_semantic: Passed to PromptGate when no gate is provided.
        """
        if gate is not None:
            self.gate = gate
        else:
            self.gate = PromptGate(skip_semantic=skip_semantic)
        self._mutator = Mutator()

    def run_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Run one sample through PromptGate and return a result record.

        Detection definition:
          - Attack "detected": actual_decision in (BLOCK, REVIEW, FLAG)
          - Attack "missed":   actual_decision == ALLOW
          - Benign "false positive": actual_decision in (BLOCK, REVIEW, FLAG)

        Layer detection is derived from the ``signals_checked`` audit strings,
        not from ``response["signals"]``. The latter is cleared to ``[]`` on
        ALLOW decisions, which would incorrectly attribute zero layers to
        near-miss attacks. ``signals_checked`` is always present and always
        records whether each layer fired, regardless of the final decision.

        Args:
            sample: Dict with keys: text, category, expected_decision, source.

        Returns:
            Result record dict with all analysis fields.
        """
        response = self.gate.check(sample["text"])

        actual = response["decision"]
        expected = sample["expected_decision"]
        is_attack = expected == "BLOCK"

        # signals_fired reads from response["signals"] — present on non-ALLOW decisions.
        # On ALLOW it is [] by design, which is correct: no signals fired above threshold.
        signals = response.get("signals", [])

        # Determine which layers fired by reading the audit strings in
        # signals_checked. These are always present regardless of decision,
        # unlike response["signals"] which is cleared to [] on ALLOW.
        signals_checked = response.get("signals_checked", [])
        layers_fired: list[str] = []
        for entry in signals_checked:
            if entry.startswith("rule_based:") and "no injection" not in entry:
                layers_fired.append("rule_based")
            elif entry.startswith("semantic:") and (
                "similar attack found" in entry
            ):
                layers_fired.append("semantic")
            elif entry.startswith("intent:") and (
                "injection intent detected" in entry
            ):
                layers_fired.append("intent")

        detected = actual in ("BLOCK", "REVIEW", "FLAG")

        return {
            "text":              sample["text"],
            "category":          sample["category"],
            "expected_decision": expected,
            "actual_decision":   actual,
            "confidence":        response["confidence"],
            "signals_fired":     [s["signal"] for s in signals],
            "layers_fired":      layers_fired,
            "correct":           actual == expected,
            "is_false_positive": not is_attack and detected,
            "is_false_negative": is_attack and not detected,
            "source":            sample.get("source", "unknown"),
        }

    def run_dataset(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run all samples through PromptGate and return result records.

        Prints a simple progress line before starting. Does not print
        per-sample progress to keep output clean.

        Args:
            samples: List of sample dicts to evaluate.

        Returns:
            List of result records in the same order as input.
        """
        total = len(samples)
        print(f"Running {total} samples...")
        return [self.run_sample(s) for s in samples]

    def run_mutations(
        self,
        text: str,
        category: str,
        methods: list[str],
        n: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate mutations of text and run each through PromptGate.

        Args:
            text: Original attack string.
            category: Attack category.
            methods: Mutation methods to apply.
            n: Number of variants for paraphrase method.

        Returns:
            List of result records. Each record's source field notes
            which mutation method produced it.
        """
        variants = self._mutator.generate(text, category=category, methods=methods, n=n)
        results: list[dict[str, Any]] = []

        for v in variants:
            sample = {
                "text":              v["text"],
                "category":          v["category"],
                "expected_decision": "BLOCK",  # all mutations of attacks are attacks
                "source":            f"mutation:{v['method']}",
            }
            record = self.run_sample(sample)
            record["mutation_method"] = v["method"]
            record["original_text"] = v["original"]
            results.append(record)

        return results