"""
tune_thresholds.py — Find the F1-optimal threshold per tactic on the val split.

For each of the 7 tactics, sweep thresholds 0.20 → 0.80 in 0.05 steps and
pick the one that maximizes that tactic's F1. Save to thresholds.json.

Run: python train/tune_thresholds.py --data data/labeled.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TACTIC_COLUMNS = [
    "t1_urgency", "t2_authority", "t3_financial",
    "t4_credential", "t5_payment", "t6_secrecy", "t7_tech_support",
]
TACTIC_NAMES = [
    "urgency", "authority", "financial_lure", "credential_harvesting",
    "payment_redirection", "secrecy", "tech_support",
]


def sweep_thresholds(probs: np.ndarray, labels: np.ndarray):
    """probs, labels: (N, 7). Returns dict tactic->best threshold and f1."""
    results = {}
    grid = np.arange(0.20, 0.81, 0.05)
    for i, name in enumerate(TACTIC_NAMES):
        best_t, best_f1 = 0.5, 0.0
        for t in grid:
            preds = (probs[:, i] >= t).astype(int)
            tp = int(((preds == 1) & (labels[:, i] == 1)).sum())
            fp = int(((preds == 1) & (labels[:, i] == 0)).sum())
            fn = int(((preds == 0) & (labels[:, i] == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        results[name] = {"threshold": best_t, "f1": round(best_f1, 4)}
        print(f"  {name:25s} best_threshold={best_t:.2f}  f1={best_f1:.4f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="models/tactic-classifier")
    p.add_argument("--out", default="models/tactic-classifier/thresholds.json")
    args = p.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text"])
    # Evaluation protocol (see README "Evaluation protocol"):
    # the val pool is split in half — thresholds are tuned on the tuning
    # half; the other half is the held-out test split evaluate.py measures.
    _, val_pool = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    val_df, _held_out_test = train_test_split(
        val_pool, test_size=0.5, random_state=99, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    texts = val_df["text"].tolist()
    labels = val_df[TACTIC_COLUMNS].values

    all_probs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 32), desc="scoring", unit="batch"):
            batch = texts[i:i + 32]
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=256, return_tensors="pt").to(device)
            logits = model(**enc).logits
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    print(f"[OK] Scored {len(texts)} val examples on {device}")

    results = sweep_thresholds(probs, labels)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Saved thresholds to {args.out}")


if __name__ == "__main__":
    main()
