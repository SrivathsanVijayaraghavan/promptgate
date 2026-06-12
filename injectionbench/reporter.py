"""
injectionbench/reporter.py
--------------------------
Generates benchmark reports from MetricsScorer output.

Produces two formats:
  - Human-readable text report (for README, terminal output)
  - JSON report (for programmatic use, archiving)

Both are saved to the results/ directory with date-stamped filenames.
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any


class BenchmarkReporter:
    """Generate and save benchmark reports from metrics dicts."""

    def generate_text_report(
        self, metrics: dict[str, Any], version: str = "unknown"
    ) -> str:
        """Generate a human-readable benchmark report.

        Args:
            metrics: Output from MetricsScorer.score().
            version: PromptGate version string for the report header.

        Returns:
            Formatted text report as a string.
        """
        s   = metrics["summary"]
        bcat = metrics["by_category"]
        blyr = metrics["by_layer"]
        missed = metrics.get("missed_samples", [])

        today = date.today().isoformat()
        total_attacks = s["total_attacks"]

        lines: list[str] = []
        lines += [
            "PromptGate Benchmark Report",
            "===========================",
            f"Version:  {version}",
            f"Date:     {today}",
            "Dataset:  deepset/prompt-injections (English-only) + manual seeds",
            "Note:     HuggingFace categories are approximate (binary-labeled source).",
            "",
            "SUMMARY",
            "-------",
            f"Total samples:       {s['total_samples']:>6}",
            f"Attack samples:      {s['total_attacks']:>6}",
            f"Benign samples:      {s['total_benign']:>6}",
            f"Overall detection:   {s['overall_detection_rate'] * 100:>5.1f}%",
            f"False positive rate: {s['false_positive_rate'] * 100:>5.1f}%",
            f"False negative rate: {s['false_negative_rate'] * 100:>5.1f}%",
            "",
            "BY CATEGORY",
            "-----------",
        ]

        for cat, data in sorted(bcat.items()):
            dr  = data["detection_rate"] * 100
            det = data["detected"]
            tot = data["total"]
            lines.append(f"  {cat:<28} {dr:>5.1f}%  ({det}/{tot})")

        lines += [
            "",
            "BY LAYER",
            "--------",
        ]

        rule_n = blyr["rule_based_only"]
        sem_n  = blyr["semantic_only"]
        both_n = blyr["both"]
        miss_n = blyr["neither"]

        def pct(n: int) -> str:
            return f"{n / total_attacks * 100:.1f}%" if total_attacks else "0.0%"

        lines += [
            f"  rule_based only:   {pct(rule_n):>6}  ({rule_n} attacks)",
            f"  semantic only:     {pct(sem_n):>6}  ({sem_n} attacks)",
            f"  both layers:       {pct(both_n):>6}  ({both_n} attacks)",
            f"  neither (missed):  {pct(miss_n):>6}  ({miss_n} attacks)",
            "",
        ]

        if missed:
            lines += [
                f"MISSED SAMPLES (first {min(10, len(missed))})",
                "-" * 30,
            ]
            for m in missed[:10]:
                text_preview = m["text"][:70] + ("..." if len(m["text"]) > 70 else "")
                lines.append(
                    f"  [{m['category']}] \"{text_preview}\" (conf: {m['confidence']:.2f})"
                )
        else:
            lines.append("MISSED SAMPLES: none — 100% detection rate.")

        lines.append("")
        return "\n".join(lines)

    def generate_json_report(
        self, metrics: dict[str, Any], version: str = "unknown"
    ) -> dict[str, Any]:
        """Return metrics dict with metadata fields added.

        Args:
            metrics: Output from MetricsScorer.score().
            version: PromptGate version string.

        Returns:
            Metrics dict with added: version, date, promptgate_version.
        """
        return {
            "promptgate_version": version,
            "date":               date.today().isoformat(),
            "dataset":            "deepset/prompt-injections + manual_seed",
            **metrics,
        }

    def save(
        self,
        metrics: dict[str, Any],
        output_dir: str = "results/",
        version: str = "unknown",
    ) -> None:
        """Save both text and JSON reports to output_dir.

        Filenames: benchmark_YYYY-MM-DD.txt and benchmark_YYYY-MM-DD.json
        Creates output_dir if it does not exist.

        Args:
            metrics: Output from MetricsScorer.score().
            output_dir: Directory to write reports into.
            version: PromptGate version string.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        today = date.today().isoformat()
        txt_path  = out / f"benchmark_{today}.txt"
        json_path = out / f"benchmark_{today}.json"

        text_report = self.generate_text_report(metrics, version=version)
        txt_path.write_text(text_report, encoding="utf-8")

        json_report = self.generate_json_report(metrics, version=version)
        json_path.write_text(
            json.dumps(json_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"Reports saved:")
        print(f"  Text: {txt_path}")
        print(f"  JSON: {json_path}")