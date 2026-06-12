"""
injectionbench/__main__.py
--------------------------
CLI entry point for InjectionBench.

Usage:
    python -m injectionbench run
    python -m injectionbench run --source manual
    python -m injectionbench run --category direct_injection
    python -m injectionbench run --mutations
    python -m injectionbench run --skip-semantic --output my_results/
"""

import argparse
import sys

from injectionbench.dataset import DatasetLoader
from injectionbench.runner import BenchmarkRunner
from injectionbench.scorer import MetricsScorer
from injectionbench.reporter import BenchmarkReporter


def main() -> None:
    """Entry point for python -m injectionbench."""
    parser = argparse.ArgumentParser(
        prog="injectionbench",
        description="Adversarial benchmarking framework for PromptGate.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the benchmark.")
    run_parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip semantic detection layer.",
    )
    run_parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Run only this attack category (manual source only).",
    )
    run_parser.add_argument(
        "--mutations",
        action="store_true",
        help="Also run mutation variants of attack samples.",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for reports (default: results/).",
    )
    run_parser.add_argument(
        "--source",
        type=str,
        default="huggingface",
        choices=["huggingface", "manual", "combined"],
        help="Dataset source (default: huggingface).",
    )

    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        sys.exit(0)

    loader   = DatasetLoader()
    runner   = BenchmarkRunner(skip_semantic=args.skip_semantic)
    scorer   = MetricsScorer()
    reporter = BenchmarkReporter()

    print(f"InjectionBench — source: {args.source}")
    print("-" * 40)

    # Load dataset
    if args.category and args.source != "manual":
        print("Note: --category filter only applies with --source manual. Switching to manual.")
        args.source = "manual"

    data = loader.load_all(source=args.source)
    attacks = data["attacks"]
    benign  = data["benign"]

    if args.category:
        attacks = [s for s in attacks if s["category"] == args.category]
        print(f"Filtered to category: {args.category} ({len(attacks)} samples)")

    print(f"Loaded {len(attacks)} attack samples, {len(benign)} benign samples.")
    print()

    # Run main dataset
    attack_results = runner.run_dataset(attacks)
    benign_results = runner.run_dataset(benign)
    all_results    = attack_results + benign_results

    # Optionally run mutations
    if args.mutations:
        print("\nRunning mutation variants...")
        mutation_results: list[dict] = []
        methods = ["case_flip", "whitespace_inject", "homoglyph"]
        for sample in attacks[:10]:  # limit to first 10 to keep it fast
            mut = runner.run_mutations(
                sample["text"],
                category=sample["category"],
                methods=methods,
            )
            mutation_results.extend(mut)
        print(f"Mutation samples run: {len(mutation_results)}")
        all_results.extend(mutation_results)

    # Score
    metrics = scorer.score(all_results)

    # Print report to terminal
    text_report = reporter.generate_text_report(metrics, version="0.2.0")
    print()
    print(text_report)

    # Save to disk
    reporter.save(metrics, output_dir=args.output, version="0.2.0")


if __name__ == "__main__":
    main()