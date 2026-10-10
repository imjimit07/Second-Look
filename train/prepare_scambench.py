"""
prepare_scambench.py — Download ScamBench, extract messages, map
scenario_category -> 7-tactic schema, and save a labeled CSV.

Uses the canonical parquet splits (data/train/*.parquet) — the
formats/eliza-*.jsonl variants use an agent-conversation schema without
language/scenario fields at top level.

Mapping rules:
- language == "en" only.
- should_trigger_scam_defense == False  -> negative example (all-zero tactics),
  regardless of category name.
- should_trigger_scam_defense == True   -> tactics from CATEGORY_TO_TACTICS;
  categories with no mapping (e.g. agent-security attacks such as
  'research-assisted') are SKIPPED, not labeled all-zero — labeling attack
  text as "no tactics" would teach the classifier the wrong thing.

Run: python train/prepare_scambench.py
Output: data/labeled.csv  (columns: text, should_trigger, scenario_category,
                            t1_urgency ... t7_tech_support)
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

TACTIC_COLUMNS = [
    "t1_urgency",
    "t2_authority",
    "t3_financial",
    "t4_credential",
    "t5_payment",
    "t6_secrecy",
    "t7_tech_support",
]

# Map ScamBench scenario_category -> 7 tactics (multi-label).
# Built from the actual English category distribution in the train split.
CATEGORY_TO_TACTICS = {
    # --- core categories (dataset card) ---
    "social-engineering":        ["t1_urgency", "t2_authority", "t6_secrecy"],
    "phishing-link":             ["t4_credential", "t1_urgency"],
    "credential-theft":          ["t4_credential", "t2_authority"],
    "impersonation":             ["t2_authority"],
    "advance-fee-fraud":         ["t3_financial", "t5_payment"],
    "secret-exfiltration":       ["t6_secrecy", "t4_credential"],
    "prompt-injection":          ["t1_urgency", "t2_authority"],
    "extended-multi-turn":       ["t1_urgency", "t2_authority"],
    "inter-agent-communication": ["t2_authority"],
    "interpersonal-abuse":       ["t1_urgency", "t6_secrecy"],
    "malicious-tool":            ["t7_tech_support"],
    # --- additional English scam categories present in the split ---
    "government-impersonation":  ["t2_authority"],
    "credential-phishing":       ["t4_credential"],
    "investment-scam":           ["t3_financial", "t5_payment"],
    "investment-fraud":          ["t3_financial", "t5_payment"],
    "crypto-investment-scam":    ["t3_financial", "t5_payment"],
    "malware-delivery":          ["t7_tech_support"],
    "social-manipulation":       ["t1_urgency", "t6_secrecy"],
    "unauthorized-access":       ["t4_credential", "t7_tech_support"],
    "phishing":                  ["t4_credential", "t1_urgency"],
    "payment-fraud":             ["t5_payment", "t3_financial"],
    "impersonation-fraud":       ["t2_authority"],
    "smishing":                  ["t1_urgency", "t4_credential"],
    "romance-scam":              ["t6_secrecy", "t5_payment"],
    "bank-phishing":             ["t4_credential", "t2_authority"],
    "fake-delivery-notification": ["t1_urgency", "t4_credential"],
    "job-scam":                  ["t3_financial", "t5_payment"],
    "loan-fraud":                ["t3_financial", "t5_payment"],
    "shopping-fraud":            ["t3_financial", "t5_payment"],
    "fake-billing":              ["t3_financial", "t5_payment"],
    "tech-support-scam":         ["t7_tech_support"],
    "voice-phishing":            ["t2_authority", "t1_urgency"],
    "extortion":                 ["t1_urgency", "t6_secrecy"],
    "dsi-impersonation":         ["t2_authority"],
    "promptpay-fraud":           ["t5_payment"],
    "crypto":                    ["t3_financial", "t5_payment"],
}


def extract_text_from_messages(messages_json: str) -> str:
    """Flatten a ScamBench messages JSON string into a single text block.

    Only user/assistant turns — the scene-setting system prompt leaks
    category information and must not be a training feature.
    """
    try:
        messages = json.loads(messages_json)
    except (json.JSONDecodeError, TypeError):
        return ""

    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def extra_negative_rows(n_sms_ham: int = 800, seed: int = 42) -> list:
    """Benign negatives covering what ScamBench's legit class lacks: one-shot
    service notifications (delivery, bank, 2FA) with real-brand URLs, plus a
    sample of everyday SMS ham. All are all-zero tactic labels.

    Without these, 'brand + a link' superficially resembles the phishing-link
    class and the classifier false-positives on delivery notifications.
    """
    rows = []
    bn = Path("data/raw/benign_notifications.csv")
    if bn.exists():
        for t in pd.read_csv(bn)["text"].dropna().tolist():
            rows.append({"text": t, "category": "template-negative"})
    else:
        print(f"[!] {bn} not found — skipping template negatives")

    sms = Path("data/raw/sms_spam.csv")
    if sms.exists():
        ham_df = pd.read_csv(sms, encoding="latin-1")
        ham = ham_df[ham_df["v1"] == "ham"]["v2"].dropna()
        ham = ham.sample(n=min(n_sms_ham, len(ham)), random_state=seed)
        rows += [{"text": t, "category": "sms-ham"} for t in ham.tolist()]
    else:
        print(f"[!] {sms} not found — skipping SMS ham")

    return rows


def build_labeled_csv(output_path: str = "data/labeled.csv"):
    Path("data").mkdir(exist_ok=True)

    print("[...] Loading ScamBench (canonical parquet train split)...")
    ds = load_dataset(
        "shaw/scambench-training",
        data_files={"train": "data/train/train-00000-of-00001.parquet"},
    )
    print(f"[OK] Loaded {len(ds['train'])} records")

    records = []
    skipped_category = Counter()
    n_short = 0

    for row in tqdm(ds["train"], desc="Processing"):
        if row.get("language") != "en":
            continue  # English only for the classifier

        category = (row.get("scenario_category") or "").strip().lower()

        text = extract_text_from_messages(row.get("messages", ""))
        if len(text) < 40:
            n_short += 1
            continue

        if not row.get("should_trigger_scam_defense"):
            tactics = []  # true negative (legitimate / benign / inquiries)
        elif category in CATEGORY_TO_TACTICS:
            tactics = CATEGORY_TO_TACTICS[category]
        else:
            skipped_category[category] += 1  # attack outside the tactic schema
            continue

        label_row = {
            "text": text,
            "should_trigger": bool(row.get("should_trigger_scam_defense")),
            "scenario_category": category,
        }
        for col in TACTIC_COLUMNS:
            label_row[col] = 1 if col in tactics else 0

        records.append(label_row)

    # Merge extra benign negatives (see extra_negative_rows docstring)
    extra = extra_negative_rows()
    for r in extra:
        label_row = {
            "text": r["text"],
            "should_trigger": False,
            "scenario_category": r["category"],
        }
        for col in TACTIC_COLUMNS:
            label_row[col] = 0
        records.append(label_row)
    print(f"[OK] Added {len(extra)} extra benign negatives")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"\n[OK] Kept {len(df)} records | too short: {n_short} | "
          f"unmapped attack categories skipped: {sum(skipped_category.values())}")
    if skipped_category:
        print("     (out-of-scope attacks:", dict(skipped_category.most_common(8)), "...)")

    print("\nBinary distribution (should_trigger):")
    print(df["should_trigger"].value_counts().to_string())

    print("\nLabel distribution:")
    for col in TACTIC_COLUMNS:
        pos = int(df[col].sum())
        pct = 100 * pos / len(df) if len(df) else 0
        print(f"  {col:20s} {pos:5d} ({pct:5.1f}%)")

    df.to_csv(output_path, index=False)
    print(f"\n[OK] Saved to {output_path}")


if __name__ == "__main__":
    build_labeled_csv()
