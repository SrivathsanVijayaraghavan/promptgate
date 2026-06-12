"""Tests for injectionbench/scorer.py"""

from injectionbench.scorer import MetricsScorer

scorer = MetricsScorer()


def _make_result(expected, actual, category="direct_injection", layers=None):
    detected = actual in ("BLOCK", "REVIEW", "FLAG")
    is_attack = expected == "BLOCK"
    return {
        "text":              "sample text",
        "category":          category,
        "expected_decision": expected,
        "actual_decision":   actual,
        "confidence":        0.9 if detected else 0.1,
        "signals_fired":     ["instruction_override"] if detected else [],
        "layers_fired":      layers if layers is not None else (["rule_based"] if detected else []),
        "correct":           expected == actual,
        "is_false_positive": not is_attack and detected,
        "is_false_negative": is_attack and not detected,
        "source":            "test",
    }


def test_overall_detection_rate_all_detected():
    results = [_make_result("BLOCK", "BLOCK") for _ in range(5)]
    metrics = scorer.score(results)
    assert metrics["summary"]["overall_detection_rate"] == 1.0


def test_overall_detection_rate_none_detected():
    results = [_make_result("BLOCK", "ALLOW") for _ in range(5)]
    metrics = scorer.score(results)
    assert metrics["summary"]["overall_detection_rate"] == 0.0


def test_overall_detection_rate_partial():
    results = (
        [_make_result("BLOCK", "BLOCK") for _ in range(3)] +
        [_make_result("BLOCK", "ALLOW") for _ in range(1)]
    )
    metrics = scorer.score(results)
    assert metrics["summary"]["overall_detection_rate"] == 0.75


def test_false_positive_rate():
    results = (
        [_make_result("ALLOW", "FLAG") for _ in range(2)] +
        [_make_result("ALLOW", "ALLOW") for _ in range(8)]
    )
    metrics = scorer.score(results)
    assert metrics["summary"]["false_positive_rate"] == 0.2


def test_false_negative_rate():
    results = (
        [_make_result("BLOCK", "ALLOW") for _ in range(2)] +
        [_make_result("BLOCK", "BLOCK") for _ in range(8)]
    )
    metrics = scorer.score(results)
    assert metrics["summary"]["false_negative_rate"] == 0.2


def test_by_category_breakdown():
    results = (
        [_make_result("BLOCK", "BLOCK", category="direct_injection") for _ in range(3)] +
        [_make_result("BLOCK", "ALLOW", category="jailbreak") for _ in range(2)]
    )
    metrics = scorer.score(results)
    assert "direct_injection" in metrics["by_category"]
    assert "jailbreak" in metrics["by_category"]
    assert metrics["by_category"]["direct_injection"]["detection_rate"] == 1.0
    assert metrics["by_category"]["jailbreak"]["detection_rate"] == 0.0


def test_by_layer_keys_present():
    results = [_make_result("BLOCK", "BLOCK")]
    metrics = scorer.score(results)
    keys = {"rule_based_only", "semantic_only", "both", "neither"}
    assert keys.issubset(metrics["by_layer"].keys())


def test_by_layer_neither_counts_missed():
    results = [_make_result("BLOCK", "ALLOW", layers=[])]
    metrics = scorer.score(results)
    assert metrics["by_layer"]["neither"] == 1


def test_by_layer_rule_only():
    results = [_make_result("BLOCK", "BLOCK", layers=["rule_based"])]
    metrics = scorer.score(results)
    assert metrics["by_layer"]["rule_based_only"] == 1


def test_by_layer_both():
    results = [_make_result("BLOCK", "BLOCK", layers=["rule_based", "semantic"])]
    metrics = scorer.score(results)
    assert metrics["by_layer"]["both"] == 1


def test_missed_samples_only_contains_missed_attacks():
    results = (
        [_make_result("BLOCK", "ALLOW") for _ in range(2)] +
        [_make_result("BLOCK", "BLOCK") for _ in range(3)] +
        [_make_result("ALLOW", "ALLOW") for _ in range(2)]
    )
    metrics = scorer.score(results)
    assert len(metrics["missed_samples"]) == 2


def test_missed_samples_have_required_fields():
    results = [_make_result("BLOCK", "ALLOW")]
    metrics = scorer.score(results)
    for m in metrics["missed_samples"]:
        assert "text" in m
        assert "category" in m
        assert "confidence" in m


def test_all_rates_are_floats_in_range():
    results = (
        [_make_result("BLOCK", "BLOCK") for _ in range(4)] +
        [_make_result("ALLOW", "ALLOW") for _ in range(4)]
    )
    metrics = scorer.score(results)
    s = metrics["summary"]
    for key in ("overall_detection_rate", "false_positive_rate", "false_negative_rate"):
        assert isinstance(s[key], float)
        assert 0.0 <= s[key] <= 1.0


def test_empty_results_no_crash():
    metrics = scorer.score([])
    assert metrics["summary"]["total_samples"] == 0
    assert metrics["summary"]["overall_detection_rate"] == 0.0