"""
train_tactic_classifier.py — Fine-tune DistilBERT for multi-label tactic classification.

Input: CSV with columns [text, t1_urgency, t2_authority, t3_financial,
                        t4_credential, t5_payment, t6_secrecy, t7_tech_support]
Output: models/tactic-classifier/

Run: python train/train_tactic_classifier.py --data data/labeled.csv
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

TACTIC_COLUMNS = [
    "t1_urgency",
    "t2_authority",
    "t3_financial",
    "t4_credential",
    "t5_payment",
    "t6_secrecy",
    "t7_tech_support",
]

TACTIC_NAMES = [
    "urgency",
    "authority",
    "financial_lure",
    "credential_harvesting",
    "payment_redirection",
    "secrecy",
    "tech_support",
]

MODEL_NAME = "distilbert-base-uncased"


class TacticDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }


def upsample_rare_tactics(train_df, min_positives=300, max_copies=5):
    """Duplicate train rows carrying rare tactics so every tactic has at
    least `min_positives` train examples (capped at `max_copies` extra
    copies per row). Applied AFTER the val split — never to val data.
    """
    out = train_df
    for col in TACTIC_COLUMNS:
        pos = out[out[col] == 1]
        n = len(pos)
        if n == 0 or n >= min_positives:
            continue
        copies = min(max_copies, math.ceil(min_positives / n) - 1)
        if copies > 0:
            out = pd.concat([out] + [pos] * copies, ignore_index=True)
            print(f"  [upsample] {col}: {n} -> {int((out[col] == 1).sum())} positives")
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    parser.add_argument("--output", default="models/tactic-classifier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--upsample_min_positives", type=int, default=300,
                        help="Upsample train rows until each rare tactic has this many positives (0=off)")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    missing = [c for c in TACTIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column")

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    print(f"[OK] Loaded {len(df)} examples")

    # Report label distribution — important for the README
    print("\nLabel distribution:")
    for col, name in zip(TACTIC_COLUMNS, TACTIC_NAMES):
        pos = int(df[col].sum())
        pct = 100 * pos / len(df)
        print(f"  {name:25s} {pos:4d} ({pct:5.1f}%)")

    # Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"\n[OK] Train: {len(train_df)} | Val: {len(val_df)}")

    # Upsample rare tactics in the TRAIN split only (val stays untouched)
    if args.upsample_min_positives > 0:
        train_df = upsample_rare_tactics(
            train_df, min_positives=args.upsample_min_positives
        )
        print(f"[OK] Train after upsampling: {len(train_df)}")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TACTIC_COLUMNS),
        problem_type="multi_label_classification",
    )

    train_ds = TacticDataset(
        train_df["text"].tolist(),
        train_df[TACTIC_COLUMNS].values,
        tokenizer,
    )
    val_ds = TacticDataset(
        val_df["text"].tolist(),
        val_df[TACTIC_COLUMNS].values,
        tokenizer,
    )

    # Training
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # Save label metadata for inference
    metadata = {
        "tactic_columns": TACTIC_COLUMNS,
        "tactic_names": TACTIC_NAMES,
        "base_model": MODEL_NAME,
        "threshold": 0.5,
    }
    with open(Path(args.output) / "label_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Final eval
    print("\n=== Final Validation Metrics ===")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"  {k}: {v:.4f}")

    print(f"\n[OK] Model saved to {args.output}")


if __name__ == "__main__":
    main()
