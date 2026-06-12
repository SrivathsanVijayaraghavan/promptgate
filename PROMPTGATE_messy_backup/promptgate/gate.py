"""Main PromptGate middleware entry point."""

from promptgate.aggregator import aggregate
from promptgate.detector.intent import IntentClassifier
from promptgate.detector.rule_based import RuleBasedDetector
from promptgate.detector.semantic import SemanticDetector
from promptgate.parser.input_parser import parse_input
from promptgate.policy import evaluate
from promptgate.response import build_response
from promptgate.scorer import score


class PromptGate:
    """AI security middleware that classifies prompt injection risk before LLM access.

    Runs a three-layer detection pipeline:
      1. Rule-based — fast keyword/phrase matching against pattern files.
      2. Semantic   — sentence-embedding similarity against known attack library.
      3. Intent     — fine-tuned DistilBERT classifier for implicit/conversational
                      injections that bypass vocabulary-based detection entirely.

    All layers feed signals into the same accumulation model. Each layer degrades
    gracefully when its optional dependencies are not installed.
    """

    def __init__(
        self,
        thresholds: dict | None = None,
        skip_semantic: bool = False,
        skip_intent: bool = False,
        semantic_threshold: float = 0.65,
        intent_threshold: float = 0.70,
    ) -> None:
        """Initialise PromptGate with optional configuration.

        Args:
            thresholds: Optional dict to override DEFAULT_THRESHOLDS.
                        Accepted keys: block, review, flag.
                        Unspecified keys fall back to DEFAULT_THRESHOLDS.
            skip_semantic: If True, the semantic detector is never called
                           regardless of whether it is installed.
            skip_intent: If True, the intent classifier is never called
                         regardless of whether it is installed.
            semantic_threshold: Cosine similarity cutoff passed to
                                SemanticDetector. Default 0.65.
            intent_threshold: INJECTION probability cutoff passed to
                              IntentClassifier. Default 0.70.
        """
        self.thresholds = thresholds
        self.skip_semantic = skip_semantic
        self.skip_intent = skip_intent
        self.rule_detector = RuleBasedDetector()
        self.semantic_detector = SemanticDetector(threshold=semantic_threshold)
        self.intent_detector = IntentClassifier(threshold=intent_threshold)

    def check(self, user_input: str) -> dict:
        """Run the full three-layer risk classification pipeline.

        Pipeline steps:
          1. parse    — normalise, lowercase, detect encoding anomalies
          2. rule     — keyword/phrase matching against pattern files
          3. semantic — embedding similarity against known attack library
          4. intent   — fine-tuned classifier for implicit/conversational attacks
          5. aggregate — group signals into threat categories
          6. score    — accumulate severities, clamp to [0.0, 1.0]
          7. decide   — map score to ALLOW / FLAG / REVIEW / BLOCK
          8. build    — assemble the final structured response dict

        Args:
            user_input: Raw user prompt text.

        Returns:
            Structured response dict with exactly 7 keys:
            decision, confidence, risk_level, threat_categories,
            signals, signals_checked, message.
        """
        # 1. Parse
        parsed = parse_input(user_input)
        cleaned = parsed["cleaned_text"]

        # 2. Rule-based detection
        rule_signals = self.rule_detector.detect(cleaned)
        rule_checked = (
            f"rule_based: {len(rule_signals)} pattern{'s' if len(rule_signals) != 1 else ''} matched"
            if rule_signals
            else "rule_based: no injection patterns found"
        )

        # 3. Semantic detection
        semantic_signals = []
        if self.skip_semantic:
            semantic_checked = "semantic: skipped by configuration"
        elif not self.semantic_detector.is_available():
            semantic_checked = "semantic: skipped (not installed)"
        else:
            semantic_signals = self.semantic_detector.detect(cleaned)
            semantic_checked = (
                "semantic: similar attack found above threshold"
                if semantic_signals
                else "semantic: no similar attacks found"
            )

        # 4. Intent classification
        intent_signals = []
        if self.skip_intent:
            intent_checked = "intent: skipped by configuration"
        elif not self.intent_detector.is_available():
            intent_checked = "intent: skipped (model not trained or not installed)"
        else:
            intent_signals = self.intent_detector.detect(cleaned)
            intent_checked = (
                "intent: injection intent detected above threshold"
                if intent_signals
                else "intent: no injection intent detected"
            )

        # 5. Merge signals from all layers
        all_signals = rule_signals + semantic_signals + intent_signals
        signals_checked = [rule_checked, semantic_checked, intent_checked]

        # 6. Aggregate into threat categories
        aggregated = aggregate(all_signals)
        signals = aggregated["signals"]
        threat_categories = aggregated["threat_categories"]

        # 7. Score and decide
        risk_score = score(signals)
        decision = evaluate(risk_score, self.thresholds)

        # 8. Build response
        return build_response(
            decision=decision,
            risk_score=risk_score,
            threat_categories=threat_categories,
            signals=signals,
            signals_checked=signals_checked,
        )