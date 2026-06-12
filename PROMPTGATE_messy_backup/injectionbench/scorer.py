"""
injectionbench/scorer.py
------------------------
Calculates benchmark metrics from BenchmarkRunner result records.

Metrics produced:
  - Summary: detection rate, false positive rate, false negative rate
  - By category: per-category detection breakdown
  - By layer: which detection layer(s) caught each attack
  - Missed samples: list of attacks that were not detected
"""

from typing import Any


_DETECTED_DECISIONS = {"BLOCK", "REVIEW", "FLAG"}


class MetricsScorer:
    """Calculate benchmark metrics from a list of runner result records.

    Detection definition used throughout:
      Attack detected    — actual_decision in (BLOCK, REVIEW, FLAG)
      Attack missed      — actual_decision == ALLOW
      Benign FP          — actual_decision in (BLOCK, REVIEW, FLAG)
    """

    def score(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate full benchmark metrics from runner result records.

        Args:
            results: List of result dicts from BenchmarkRunner.run_sample()
                     or run_dataset(). May contain both attack and benign results.

        Returns:
            Metrics dict with keys: summary, by_category, by_layer,
            missed_samples.
        """
        attacks = [r for r in results if r["expected_decision"] == "BLOCK"]
        benign  = [r for r in results if r["expected_decision"] == "ALLOW"]

        total_attacks = len(attacks)
        total_benign  = len(benign)

        detected_attacks = [r for r in attacks if r["actual_decision"] in _DETECTED_DECISIONS]
        missed_attacks   = [r for r in attacks if r["actual_decision"] not in _DETECTED_DECISIONS]
        false_positives  = [r for r in benign  if r["actual_decision"] in _DETECTED_DECISIONS]

        overall_dr  = len(detected_attacks) / total_attacks if total_attacks else 0.0
        fp_rate     = len(false_positives)  / total_benign  if total_benign  else 0.0
        fn_rate     = len(missed_attacks)   / total_attacks if total_attacks else 0.0

        summary = {
            "total_samples":          len(results),
            "total_attacks":          total_attacks,
            "total_benign":           total_benign,
            "overall_detection_rate": round(overall_dr, 4),
            "false_positive_rate":    round(fp_rate, 4),
            "false_negative_rate":    round(fn_rate, 4),
        }

        by_category = self._score_by_category(attacks)
        by_layer    = self._score_by_layer(attacks)

        missed_samples = [
            {
                "text":       r["text"],
                "category":   r["category"],
                "confidence": r["confidence"],
            }
            for r in missed_attacks
        ]

        return {
            "summary":        summary,
            "by_category":    by_category,
            "by_layer":       by_layer,
            "missed_samples": missed_samples,
        }

    def _score_by_category(
        self, attacks: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Break down detection rate per attack category.

        Args:
            attacks: Attack-only result records.

        Returns:
            Dict mapping category name → {total, detected, missed, detection_rate}.
        """
        categories: dict[str, list] = {}
        for r in attacks:
            cat = r["category"]
            categories.setdefault(cat, []).append(r)

        result: dict[str, dict[str, Any]] = {}
        for cat, records in sorted(categories.items()):
            detected = sum(
                1 for r in records if r["actual_decision"] in _DETECTED_DECISIONS
            )
            missed = len(records) - detected
            result[cat] = {
                "total":          len(records),
                "detected":       detected,
                "missed":         missed,
                "detection_rate": round(detected / len(records), 4) if records else 0.0,
            }
        return result

    def _score_by_layer(
        self, attacks: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Count how many attacks were caught by each detection layer combination.

        Three-layer aware: rule_based, semantic, and intent.
        Each attack is assigned to exactly one bucket based on which layers fired.

        Single-layer catches are counted individually.
        Any combination of 2+ layers is counted under "both" (preserves
        backwards compatibility while correctly attributing intent catches).
        Missed attacks are counted under "neither".

        Args:
            attacks: Attack-only result records.

        Returns:
            Dict with keys: rule_based_only, semantic_only, intent_only,
            both, neither.
        """
        rule_only   = 0
        sem_only    = 0
        intent_only = 0
        both        = 0   # 2+ layers fired
        neither     = 0

        for r in attacks:
            layers = set(r.get("layers_fired", []))
            has_rule   = "rule_based" in layers
            has_sem    = "semantic"   in layers
            has_intent = "intent"     in layers

            active_count = sum([has_rule, has_sem, has_intent])

            if active_count >= 2:
                both += 1
            elif has_rule:
                rule_only += 1
            elif has_sem:
                sem_only += 1
            elif has_intent:
                intent_only += 1
            else:
                neither += 1

        return {
            "rule_based_only": rule_only,
            "semantic_only":   sem_only,
            "intent_only":     intent_only,
            "both":            both,
            "neither":         neither,
        }