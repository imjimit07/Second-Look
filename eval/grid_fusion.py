"""
grid_fusion.py — Tune fuse() decision constants on a balanced held-out sample.

Caches per-row evidence once (classifier + URL + retrieval + keywords), then
sweeps SCAM_THRESHOLD x PATTERN_PRIMARY_SCALE x CO_OCCURRENCE_BONUS_PER_TACTIC
by monkey-patching the module constants — no re-scoring per combo.

Run: python eval/grid_fusion.py
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.classify import get_classifier
from src.extract import extract_all
from src.features_url import analyze_urls
from src.retrieve import get_retriever
import src.fuse as fuse_mod

SAMPLE_PER_CLASS = 300
SEED = 7


def build_evidence(text, classifier, retriever):
    entities = extract_all(text)
    pred = classifier.predict(text)
    url_scores = []
    if entities.urls:
        try:
            url_scores = [f.risk_score for f in analyze_urls(entities.urls)]
        except Exception:
            pass
    patterns = [m.to_dict() for m in retriever.search(text, top_k=3).matches]
    return {
        "tactic_probabilities": pred.probabilities,
        "tactic_thresholds": classifier.thresholds,
        "url_risk_scores": url_scores,
        "retrieved_patterns": patterns,
        "extracted_signals": {
            "urgency": entities.urgency_signals,
            "financial": entities.financial_signals,
            "authority": entities.authority_signals,
        },
    }


def main():
    df = pd.read_csv("data/labeled.csv").dropna(subset=["text"])
    # Evaluation protocol: fusion constants are tuned on the TUNING half of
    # the val pool only — the other half is the held-out test split.
    _, val_pool = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    val_df, _held_out_test = train_test_split(
        val_pool, test_size=0.5, random_state=99, shuffle=True)
    val_df = val_df.copy()
    val_df["is_scam"] = val_df["should_trigger"].astype(bool).astype(int)

    scam = val_df[val_df.is_scam == 1].sample(
        min(SAMPLE_PER_CLASS, int(val_df.is_scam.sum())), random_state=SEED)
    legit = val_df[val_df.is_scam == 0].sample(
        min(SAMPLE_PER_CLASS, int((1 - val_df.is_scam).sum())), random_state=SEED)
    sample = pd.concat([scam, legit]).sample(frac=1.0, random_state=SEED)
    y = sample["is_scam"].tolist()
    print(f"[OK] Grid sample: {len(sample)} rows "
          f"({int(sample.is_scam.sum())} scam / {int((1 - sample.is_scam).sum())} legit)")

    clf = get_classifier()
    ret = get_retriever()
    evidences = [build_evidence(t, clf, ret)
                 for t in tqdm(sample["text"], desc="caching evidence")]
    print("[OK] Evidence cached\n")

    results = []
    for scam_t, scale, bonus in itertools.product(
            [55, 60, 65, 70], [0.4, 0.6, 0.8], [3, 5, 8]):
        fuse_mod.SCAM_THRESHOLD = scam_t
        fuse_mod.PATTERN_PRIMARY_SCALE = scale
        fuse_mod.CO_OCCURRENCE_BONUS_PER_TACTIC = bonus
        preds = [int(fuse_mod.fuse(e).is_scam) for e in evidences]
        results.append({
            "SCAM_THRESHOLD": scam_t, "PATTERN_SCALE": scale, "BONUS": bonus,
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
        })

    results.sort(key=lambda r: (r["accuracy"], r["f1"]), reverse=True)
    print(f"{'SCAM_T':>7} {'SCALE':>6} {'BONUS':>6} {'ACC':>7} {'PREC':>7} {'REC':>7} {'F1':>7}")
    for r in results[:10]:
        print(f"{r['SCAM_THRESHOLD']:>7} {r['PATTERN_SCALE']:>6} {r['BONUS']:>6} "
              f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")

    best = results[0]
    print(f"\n[OK] Winner: SCAM_THRESHOLD={best['SCAM_THRESHOLD']} "
          f"PATTERN_PRIMARY_SCALE={best['PATTERN_SCALE']} "
          f"CO_OCCURRENCE_BONUS_PER_TACTIC={best['BONUS']} "
          f"(acc={best['accuracy']:.4f}, f1={best['f1']:.4f})")


if __name__ == "__main__":
    main()
