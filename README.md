# PromptGate

PromptGate is an open-source Python middleware library that sits between a user and an LLM and **detects prompt injection risk before the model sees the input**. It is a **risk classifier**, not a moral judge: it accumulates explainable signals and applies policy thresholds.

## Architecture

```
user_input
  → parser          (normalize text, metadata)
  → rule_based      (substring pattern matching)
  → aggregator      (map signals → threat categories)
  → scorer          (sum unique signal severities, cap at 1.0)
  → policy          (ALLOW / FLAG / REVIEW / BLOCK)
  → response        (structured, explainable output)
```

The LLM never receives blocked prompts when PromptGate is placed in front of the request path.

## Installation

From the project root (`promptgate/`):

```bash
pip install -e ".[dev]"
```

Editable install keeps pattern files and source in sync during development.

## Usage

```python
from promptgate import PromptGate

gate = PromptGate()
result = gate.check("Please ignore previous instructions.")

print(result["decision"])      # BLOCK
print(result["confidence"])    # 0.95
print(result["message"])       # Human-readable explanation
print(result["signals"])       # Matched risk signals
```

Custom policy thresholds:

```python
gate = PromptGate(thresholds={"block": 0.80, "review": 0.60, "flag": 0.35})
result = gate.check(user_input)
```

## Explainability Philosophy

Every response includes:

- **decision** — ALLOW, FLAG, REVIEW, or BLOCK
- **confidence** — equals the accumulated risk score (0.0 when safe)
- **signals** — what matched, with severity and pattern text
- **signals_checked** — signal types, categories, and pattern files scanned
- **message** — plain-language explanation of why the decision was made

ALLOW responses explicitly state that no injection patterns or manipulation framing were detected above thresholds. Restricted responses name matched signals and categories.

**Signal accumulation is required.** One weak signal alone (for example, sympathy framing) does not block. Multiple signals combine via `score = min(1.0, sum(severities))`.

## Threat Categories

| Category | Example signals |
|----------|-----------------|
| `direct_injection` | instruction_override, data_exfiltration |
| `jailbreak` | jailbreak_persona |
| `system_override` | system_override, system_prompt_leak |
| `social_engineering` | authority_claim, secrecy_request, urgency_framing |
| `encoding_attack` | encoding_trick |

Severity values and mappings live in `promptgate/config.py`.

## Default Policy Thresholds

| Score range | Decision |
|-------------|----------|
| 0.00 – 0.29 | ALLOW |
| 0.30 – 0.54 | FLAG |
| 0.55 – 0.74 | REVIEW |
| 0.75 – 1.00 | BLOCK |

## Local Testing

```bash
cd promptgate
pip install -e ".[dev]"
pytest -v
```

## Project Layout

```
promptgate/
├── promptgate/       # Python package
├── data/patterns/   # Seed pattern files
├── tests/
├── pyproject.toml
└── README.md
```

## License

MIT
