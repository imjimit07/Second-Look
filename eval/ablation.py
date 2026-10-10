"""
ablation.py — Component ablation study.

Re-runs the binary scam-detection eval with each component disabled, using
the evidence cache written by eval/evaluate.py (eval/cached_evidence.jsonl),
so each ablation is instant. Measures what each component actually
contributes through the production fuse() path.

Variants:
  full        — the shipped system (all components)
  no_url      — URL risk features removed
  no_retrieval— pattern retrieval removed
  no_tactics  — tactic classifier removed (URL + retrieval only)
  fusion_max  — our fusion formula replaced by a plain max-of-components
                (no co-occurrence bonus, no corroboration)

Run: python eval/evaluate.py   (once, to build the cache)
     python eval/ablation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.fuse as fuse_mod
from src.fuse import fuse

CACHE = Path("eval/cached_evidence.jsonl")

# Deterministic-fusion overrides for the fusion_max variant
FUSION_MAX_OVERRIDES = {
    "CO_OCCURRENCE_BONUS_PER_TACTIC": 0,
    "KEYWORD_WEIGHT": 0,
    "PATTERN_WEIGHT": 0,
}


def run_variant(rows, variant: str):
    tp = fp = fn = tn = 0
    saved = {}

    if variant == "fusion_max":
        for attr, value in FUSION_MAX_OVERRIDES.items():
            saved[attr] = getattr(fuse_mod, attr)
            setattr(fuse_mod, attr, value)

    try:
        for row in rows:
            ev = dict(row["evidence"])
            truth = bool(row["truth"])

            if variant == "no_url":
                ev["url_risk_scores"] = []
            elif variant == "no_retrieval":
                ev["retrieved_patterns"] = []
            elif variant == "no_tactics":
                ev["tactic_probabilities"] = {}

            pred = fuse(ev).is_scam

            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
            else:
                tn += 1
    finally:
        for attr, value in saved.items():
            setattr(fuse_mod, attr, value)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1


def main():
    if not CACHE.exists():
        sys.exit(f"No cache at {CACHE}. Run: python eval/evaluate.py first.")

    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[OK] Loaded {len(rows)} cached evidence rows\n")

    variants = ["full", "no_url", "no_retrieval", "no_tactics", "fusion_max"]
    labels = {
        "full": "Full Second Look",
        "no_url": "- URL features",
        "no_retrieval": "- Retrieval",
        "no_tactics": "- Tactic classifier",
        "fusion_max": "- Fusion (max of components)",
    }
    print(f"{'Variant':<32} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    results = {}
    for v in variants:
        a, p, r, f = run_variant(rows, v)
        results[v] = (a, p, r, f)
        print(f"{labels[v]:<32} {a:>6.3f} {p:>6.3f} {r:>6.3f} {f:>6.3f}")

    print("\nMarkdown table:\n")
    print("| System | Accuracy | Precision | Recall | F1 |")
    print("|---|---|---|---|---|")
    for v in variants:
        a, p, r, f = results[v]
        print(f"| {labels[v]} | {a:.3f} | {p:.3f} | {r:.3f} | {f:.3f} |")


if __name__ == "__main__":
    main()
