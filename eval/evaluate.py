"""
evaluate.py — Compare Second Look against two baselines on the held-out
test split.

Evaluation protocol (README "Evaluation protocol"): the val pool carved by
train_tactic_classifier.py is split in half — thresholds (tune_thresholds.py)
and fusion constants (grid_fusion.py) are tuned on the TUNING half; this
script measures final metrics on the untouched TEST half. No evaluated
message was seen during training or parameter tuning.

Systems compared:
  1. Keyword rules baseline  — urgency/financial/authority/url signal counts
  2. Zero-shot Ollama        — ask the local LLM "is this a scam?"
  3. Second Look             — the SAME fuse() evidence-fusion path the app uses
     (tactic classifier + URL features + retrieval + keyword corroboration)

Outputs (single source of truth for the README):
  eval/results_binary.csv  — binary scam/legit metrics per system
  eval/metrics.json        — {binary: {...}, tactic_level: {...}} in one file
  eval/cached_evidence.jsonl — per-row evidence, consumed by ablation.py

Run: python eval/evaluate.py [--sample 300] [--skip-ollama]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Make src/ importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extract import extract_all  # noqa: E402
from src.features_url import analyze_urls  # noqa: E402
from src.fuse import fuse  # noqa: E402

TACTIC_COLUMNS = [
    "t1_urgency", "t2_authority", "t3_financial", "t4_credential",
    "t5_payment", "t6_secrecy", "t7_tech_support",
]
KEYWORD_MIN_SIGNALS = 2   # keyword baseline needs 2+ red-flag signals


# ---------------------------------------------------------------- data
def get_heldout_split(csv_path: str = "data/labeled.csv") -> pd.DataFrame:
    """Held-out TEST split — the half of the val pool NOT used for tuning.

    Protocol: train (80%) / tuning-val (10%) / held-out test (10%).
    tune_thresholds.py and grid_fusion.py operate on the tuning half;
    this function returns the untouched test half.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    _, val_pool = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    _tuning_val, held_out = train_test_split(
        val_pool, test_size=0.5, random_state=99, shuffle=True)
    return held_out.reset_index(drop=True)


def balanced_sample(val_df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Balanced scam/legit sample so baselines are comparable."""
    df = val_df.copy()
    df["is_scam"] = df["should_trigger"].astype(bool).astype(int)
    per_class = max(1, n // 2)
    scam = df[df.is_scam == 1].sample(
        n=min(per_class, int(df.is_scam.sum())), random_state=seed)
    legit = df[df.is_scam == 0].sample(
        n=min(per_class, int((1 - df.is_scam).sum())), random_state=seed)
    return pd.concat([scam, legit]).sample(
        frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- systems
def predict_keyword(text: str) -> int:
    """Baseline 1: scam if 2+ independent red-flag keyword signals."""
    return int(extract_all(text).risk_signal_count() >= KEYWORD_MIN_SIGNALS)


def predict_ollama(text: str, model: str) -> int:
    """Baseline 2: zero-shot local LLM judgment."""
    import ollama
    prompt = (
        "Decide whether this message is a scam/phishing attempt or a "
        "legitimate message. Answer with exactly one word: SCAM or LEGIT.\n\n"
        f"MESSAGE:\n{text[:2000]}"
    )
    try:
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 8},
        )
        answer = resp["message"]["content"].upper()
        if "SCAM" in answer:
            return 1
        if "LEGIT" in answer:
            return 0
    except Exception:
        pass
    return -1  # unparsable / unavailable — excluded from metrics


def build_evidence(text: str, classifier, retriever=None) -> dict:
    """Assemble the exact evidence dict app.py builds for fuse()."""
    entities = extract_all(text)
    pred = classifier.predict(text)

    url_scores = []
    if entities.urls:
        try:
            url_scores = [f.risk_score for f in analyze_urls(entities.urls)]
        except Exception:
            pass

    patterns = []
    if retriever is not None:
        try:
            patterns = [m.to_dict() for m in retriever.search(text, top_k=3).matches]
        except Exception:
            pass

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


def predict_second_look(text: str, classifier, retriever=None) -> int:
    """Second Look via the production fuse() path. Binary: scam = decision 'scam'
    (the 'suspicious' band counts as not-scam — conservative)."""
    return int(fuse(build_evidence(text, classifier, retriever)).is_scam)


# ---------------------------------------------------------------- metrics
def binary_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300,
                    help="Balanced eval sample size (default 300)")
    ap.add_argument("--skip-ollama", action="store_true",
                    help="Skip the zero-shot Ollama baseline")
    args = ap.parse_args()

    out_dir = Path("eval")
    out_dir.mkdir(exist_ok=True)

    print("[...] Rebuilding held-out test split from data/labeled.csv")
    val_df = get_heldout_split()
    print(f"[OK] Held-out test split: {len(val_df)} rows "
          f"({val_df['should_trigger'].astype(bool).sum()} scam)")

    df = balanced_sample(val_df, args.sample)
    print(f"[OK] Eval sample: {len(df)} rows "
          f"({int(df.is_scam.sum())} scam / {int((1 - df.is_scam).sum())} legit)\n")

    y_true = df["is_scam"].tolist()
    texts = df["text"].tolist()
    results = {}

    # --- Baseline 1: keyword rules ---
    print("[...] Baseline 1: keyword rules")
    kw_pred = [predict_keyword(t) for t in tqdm(texts, desc="keywords")]
    results["keyword_rules"] = binary_metrics(y_true, kw_pred)

    # --- Baseline 2: zero-shot Ollama ---
    if not args.skip_ollama:
        from src.explain import resolve_model
        model = resolve_model()
        print(f"[...] Baseline 2: zero-shot Ollama ({model})")
        ol_pred = [predict_ollama(t, model) for t in tqdm(texts, desc="ollama")]
        valid = [i for i, p in enumerate(ol_pred) if p != -1]
        failed = len(ol_pred) - len(valid)
        if valid:
            results["ollama_zero_shot"] = binary_metrics(
                [y_true[i] for i in valid], [ol_pred[i] for i in valid])
            results["ollama_zero_shot"]["coverage"] = round(len(valid) / len(ol_pred), 4)
        if failed:
            print(f"[!] Ollama returned unparsable/failed answers for {failed} rows "
                  f"(excluded from its metrics)")

    # --- System: Second Look ---
    print("[...] Second Look: fuse() — classifier + URL + retrieval + keywords")
    tactic_metrics = None
    try:
        from src.classify import get_classifier
        classifier = get_classifier()
        try:
            from src.retrieve import get_retriever
            retriever = get_retriever()
        except FileNotFoundError as e:
            print(f"[!] {e}\n    Continuing without retrieval evidence.")
            retriever = None
        sl_pred = []
        cache_path = out_dir / "cached_evidence.jsonl"
        with open(cache_path, "w", encoding="utf-8") as cache_f:
            for t, truth in tqdm(list(zip(texts, y_true)), desc="second-look"):
                ev = build_evidence(t, classifier, retriever)
                sl_pred.append(int(fuse(ev).is_scam))
                cache_f.write(json.dumps(
                    {"evidence": ev, "truth": int(truth)},
                    ensure_ascii=False) + "\n")
        print(f"[OK] Evidence cache saved to {cache_path}")
        results["second_look"] = binary_metrics(y_true, sl_pred)

        # Tactic-level multi-label metrics on the FULL held-out test split
        y_tac_true = val_df[TACTIC_COLUMNS].values
        y_tac_pred = []
        for t in tqdm(val_df["text"].tolist(), desc="tactic-probs"):
            pred = classifier.predict(t)
            y_tac_pred.append([
                1 if pred.probabilities[n] >= classifier.thresholds.get(n, 0.5) else 0
                for n in classifier.tactic_names
            ])
        per_tactic = {}
        for i, name in enumerate(classifier.tactic_names):
            per_tactic[name] = {
                "threshold": round(classifier.thresholds.get(name, 0.5), 2),
                "f1": round(f1_score(
                    y_tac_true[:, i], [r[i] for r in y_tac_pred],
                    zero_division=0), 4),
            }
        tactic_metrics = {
            "f1_micro": round(
                f1_score(y_tac_true, y_tac_pred, average="micro", zero_division=0), 4),
            "f1_macro": round(
                f1_score(y_tac_true, y_tac_pred, average="macro", zero_division=0), 4),
            "sample_size": len(val_df),
            "per_tactic": per_tactic,
        }
    except FileNotFoundError as e:
        print(f"[!] {e}\n    Skipping Second Look system metrics.")

    # --- Report (single source of truth: one script, one output) ---
    table = pd.DataFrame(results).T
    print("\n=== Binary scam-detection results (balanced held-out sample) ===")
    print(table.to_string())
    table.to_csv(out_dir / "results_binary.csv")

    if tactic_metrics:
        print("\n=== Tactic-level (full held-out test split, tuned thresholds) ===")
        print(f"  f1_micro={tactic_metrics['f1_micro']}  "
              f"f1_macro={tactic_metrics['f1_macro']}  "
              f"n={tactic_metrics['sample_size']}")
        for name, m in tactic_metrics["per_tactic"].items():
            print(f"    {name:25s} thr={m['threshold']:<5} f1={m['f1']}")

    metrics_out = {
        "protocol": {
            "train_size": 11680,
            "tuning_val_size": 1460,
            "held_out_test_size": int(len(val_df)),
            "note": "Thresholds and fusion constants tuned on tuning-val only; "
                    "all reported metrics measured on the held-out test split.",
        },
        "binary": {**results, "sample_size": len(df)},
        "tactic_level": tactic_metrics or {},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n[OK] Saved {out_dir / 'results_binary.csv'} and {out_dir / 'metrics.json'}")
    print("\nNote: tactic labels are proxy labels mapped from ScamBench")
    print("      scenario_category — treat tactic F1 as a weak-label estimate.")


if __name__ == "__main__":
    main()
