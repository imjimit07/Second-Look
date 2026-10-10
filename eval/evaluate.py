"""
evaluate.py — Compare Second Look against two baselines on held-out data.

Held-out set: the exact validation split that train_tactic_classifier.py
carved from data/labeled.csv (same train_test_split args), so no evaluated
message was seen during training.

Systems compared:
  1. Keyword rules baseline  — urgency/financial/authority/url signal counts
  2. Zero-shot Ollama        — ask the local LLM "is this a scam?"
  3. Second Look             — the SAME fuse() evidence-fusion path the app uses
     (tactic classifier + URL features + retrieval + keyword corroboration)

Outputs:
  eval/results_binary.csv  — precision/recall/F1 per system (binary scam/legit)
  eval/metrics.json        — binary metrics + tactic-level F1 for Second Look

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
    """Reproduce the exact val split used by train_tactic_classifier.py."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    return val_df.reset_index(drop=True)


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

    print("[...] Rebuilding held-out split from data/labeled.csv")
    val_df = get_heldout_split()
    print(f"[OK] Held-out pool: {len(val_df)} rows "
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
    try:
        from src.classify import get_classifier
        classifier = get_classifier()
        try:
            from src.retrieve import get_retriever
            retriever = get_retriever()
        except FileNotFoundError as e:
            print(f"[!] {e}\n    Continuing without retrieval evidence.")
            retriever = None
        sl_pred = [predict_second_look(t, classifier, retriever)
                   for t in tqdm(texts, desc="second-look")]
        results["second_look"] = binary_metrics(y_true, sl_pred)

        # Tactic-level multi-label metrics (proxy labels from scenario_category)
        y_tac_true = df[TACTIC_COLUMNS].values
        y_tac_pred = []
        for t in tqdm(texts, desc="tactic-probs"):
            pred = classifier.predict(t)
            y_tac_pred.append([
                1 if pred.probabilities[n] >= classifier.thresholds.get(n, 0.5) else 0
                for n in classifier.tactic_names
            ])
        results["second_look"]["tactic_f1_micro"] = round(
            f1_score(y_tac_true, y_tac_pred, average="micro", zero_division=0), 4)
        results["second_look"]["tactic_f1_macro"] = round(
            f1_score(y_tac_true, y_tac_pred, average="macro", zero_division=0), 4)
    except FileNotFoundError as e:
        print(f"[!] {e}\n    Skipping Second Look system metrics.")

    # --- Report ---
    table = pd.DataFrame(results).T
    print("\n=== Binary scam-detection results (held-out sample) ===")
    print(table.to_string())
    table.to_csv(out_dir / "results_binary.csv")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"sample_size": len(df), "systems": results}, f, indent=2)
    print(f"\n[OK] Saved {out_dir / 'results_binary.csv'} and {out_dir / 'metrics.json'}")
    print("\nNote: tactic labels are proxy labels mapped from ScamBench")
    print("      scenario_category — treat tactic F1 as a weak-label estimate.")


if __name__ == "__main__":
    main()
