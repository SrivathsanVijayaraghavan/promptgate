"""
scripts/train_intent_classifier.py
-----------------------------------
Fine-tune a DistilBERT binary classifier on deepset/prompt-injections.

Produces a saved model directory at models/intent_classifier/ which
IntentClassifier loads at runtime. Run this once before using Phase 4.

Usage:
    python scripts/train_intent_classifier.py

Requirements (install with pip install promptgate[intent]):
    transformers>=4.30.0
    datasets
    torch
    scikit-learn

Training details:
    Base model:   distilbert-base-uncased
    Task:         binary classification (INJECTION=1, BENIGN=0)
    Dataset:      deepset/prompt-injections (train + test splits combined,
                  then re-split 80/20 stratified)
    Epochs:       3
    Batch size:   16
    Max length:   128 tokens (sufficient for injection prompts)
    Output dir:   models/intent_classifier/

The script prints a classification report on the held-out eval split
before saving. Target: F1 >= 0.80 on the eval split.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so promptgate imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models" / "intent_classifier"


def main() -> None:
    # ── Dependency check ─────────────────────────────────────────────────────
    try:
        import torch
        from datasets import load_dataset
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        import numpy as np
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with:")
        print("  pip install transformers[torch] datasets scikit-learn accelerate")
        sys.exit(1)

    print("=" * 60)
    print("PromptGate — Intent Classifier Training")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading deepset/prompt-injections from HuggingFace...")
    ds = load_dataset("deepset/prompt-injections")

    # Combine train and test splits — we'll make our own split
    all_texts  = list(ds["train"]["text"])  + list(ds["test"]["text"])
    all_labels = list(ds["train"]["label"]) + list(ds["test"]["label"])

    print(f"      Total samples: {len(all_texts)}")
    print(f"      Injections:    {sum(all_labels)}")
    print(f"      Benign:        {len(all_labels) - sum(all_labels)}")

    # 80/20 stratified split
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        all_texts, all_labels,
        test_size=0.20,
        random_state=42,
        stratify=all_labels,
    )
    print(f"      Train: {len(train_texts)} | Eval: {len(eval_texts)}")

    # ── Tokenise ─────────────────────────────────────────────────────────────
    print("\n[2/5] Tokenising...")
    BASE_MODEL = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenise(texts: list[str]) -> dict:
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

    train_enc = tokenise(train_texts)
    eval_enc  = tokenise(eval_texts)

    # Build torch Dataset
    import torch

    class InjectionDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: dict, labels: list[int]) -> None:
            self.encodings = encodings
            self.labels    = labels

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_dataset = InjectionDataset(train_enc, train_labels)
    eval_dataset  = InjectionDataset(eval_enc,  eval_labels)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[3/5] Loading base model: {BASE_MODEL}")
    id2label = {0: "BENIGN", 1: "INJECTION"}
    label2id = {"BENIGN": 0, "INJECTION": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n[4/5] Fine-tuning...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(
            labels, preds,
            target_names=["BENIGN", "INJECTION"],
            output_dict=True,
            zero_division=0,
        )
        return {
            "f1_injection": report["INJECTION"]["f1-score"],
            "f1_macro":     report["macro avg"]["f1-score"],
            "accuracy":     report["accuracy"],
        }

    # `eval_strategy` was renamed from `evaluation_strategy` in transformers 4.46+
    # and then the old name was removed. Detect which one this version accepts.
    import inspect
    _trainer_params = inspect.signature(TrainingArguments.__init__).parameters
    _eval_strategy_key = (
        "eval_strategy" if "eval_strategy" in _trainer_params else "evaluation_strategy"
    )

    args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        **{_eval_strategy_key: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_injection",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── Eval report ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluation on held-out split:")
    preds_output = trainer.predict(eval_dataset)
    preds        = np.argmax(preds_output.predictions, axis=-1)

    print()
    print(classification_report(
        eval_labels, preds,
        target_names=["BENIGN", "INJECTION"],
        zero_division=0,
    ))

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"Saving model to: {MODEL_DIR}")
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print("\nDone. IntentClassifier will load from:", MODEL_DIR)
    print("Run: python -m injectionbench run --source huggingface")


if __name__ == "__main__":
    main()