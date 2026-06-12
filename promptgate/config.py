"""Configuration constants for PromptGate risk classification."""

SIGNAL_SEVERITIES = {
    # Direct injection — highest risk, single signal can approach BLOCK alone
    "instruction_override":  0.95,
    "system_prompt_leak":    0.90,
    "jailbreak_persona":     0.85,
    "system_override":       0.85,
    "encoding_trick":        0.80,
    "data_exfiltration":     0.70,

    # Social engineering — only dangerous in combination
    "authority_claim":       0.40,
    "urgency_framing":       0.35,
    "secrecy_request":       0.35,
    "sympathy_manipulation": 0.25,
    "flattery":              0.20,

    # Operation signals — contextual risk indicator
    "sensitive_operation":   0.30,

    # Semantic similarity — catches paraphrased attacks missed by rule-based
    "semantic_similarity":   0.60,

    # Intent classifier — catches implicit/conversational injections that
    # bypass vocabulary-based detection entirely. Severity is set below
    # canonical rule-based signals but above semantic similarity, reflecting
    # the classifier's specificity to injection intent.
    "intent_injection":      0.75,
}

DEFAULT_THRESHOLDS = {
    "block":  0.75,
    "review": 0.55,
    "flag":   0.30,
}

# Maps each signal type to its threat category in the response.
# system_prompt_leak has its own category per spec response examples.
SIGNAL_TO_CATEGORY = {
    "instruction_override":  "direct_injection",
    "system_prompt_leak":    "system_prompt_leak",
    "jailbreak_persona":     "jailbreak",
    "system_override":       "system_override",
    "encoding_trick":        "encoding_attack",
    "data_exfiltration":     "data_exfiltration",
    "authority_claim":       "social_engineering",
    "urgency_framing":       "social_engineering",
    "secrecy_request":       "social_engineering",
    "sympathy_manipulation": "social_engineering",
    "flattery":              "social_engineering",
    "sensitive_operation":   "operation",
    "semantic_similarity":   "semantic",
    "intent_injection":      "intent_injection",
}