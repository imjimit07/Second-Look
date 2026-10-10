"""
make_labeling_draft.py — Generate a REVIEW-REQUIRED labeling worksheet.

Samples messages from the downloaded corpora and PRE-FILLS tactic labels
using the deterministic extract.py signals. These pre-filled labels are
weak heuristics, not ground truth — review and correct every row, then
save the result as data/labeled.csv for training.

Run: python make_labeling_draft.py
"""

import pandas as pd

from src.extract import extract_all

COLS = ["t1_urgency", "t2_authority", "t3_financial", "t4_credential",
        "t5_payment", "t6_secrecy", "t7_tech_support"]

# Supplemental keyword lists for tactics extract.py doesn't cover directly
CREDENTIAL_HINTS = [
    "verify your", "login", "log in", "sign in", "password", "otp",
    "one time password", "pin number", "kyc", "card number", "ssn",
    "social security number", "confirm your identity", "bank details",
    "credentials", "account number",
]
PAYMENT_HINTS = [
    "wire", "gift card", "itunes", "bitcoin", "btc", "usdt",
    "western union", "moneygram", "upi", "bank transfer",
    "processing fee", "registration fee", "advance fee", "send money",
    "transfer the",
]
SECRECY_HINTS = [
    "confidential", "do not tell", "don't tell", "keep this secret",
    "do not contact", "don't contact", "keep it between", "discreet",
    "between us",
]
TECH_HINTS = [
    "virus", "infected", "malware", "anydesk", "teamviewer",
    "remote access", "technician", "antivirus", "defender",
]

MAX_TEXT_CHARS = 1200  # ~256 BERT tokens; long bodies get truncated anyway


def weak_labels(text: str) -> dict:
    """Heuristic pre-fill. False positives and misses are EXPECTED."""
    ent = extract_all(text)
    low = text.lower()

    def hit(words):
        return any(w in low for w in words)

    labels = {
        "t1_urgency": int(bool(ent.urgency_signals)),
        "t2_authority": int(bool(ent.authority_signals or ent.brands_mentioned)),
        "t3_financial": int(bool(ent.financial_signals or ent.amounts)),
        "t4_credential": int(hit(CREDENTIAL_HINTS)),
        "t5_payment": int(hit(PAYMENT_HINTS)),
        "t6_secrecy": int(hit(SECRECY_HINTS)),
        "t7_tech_support": int(hit(TECH_HINTS)),
    }
    return labels


def clean(text: str) -> str:
    return " ".join(str(text).split())[:MAX_TEXT_CHARS]


def main():
    rows = []

    # Nazario phishing emails (label=1, all phishing)
    naz = pd.read_csv("data/raw/nazario.csv").dropna(subset=["body"])
    naz["body"] = naz["body"].map(clean)
    naz = naz[naz["body"].str.len() >= 30].drop_duplicates(subset=["body"])
    for i, body in enumerate(naz.sample(min(200, len(naz)), random_state=42)["body"]):
        rows.append({"message_id": f"n{i:03d}", "text": body,
                     **weak_labels(body), "notes": "draft: nazario phishing — verify labels"})

    # SMS spam (mixed scam types)
    sms = pd.read_csv("data/raw/sms_spam.csv", encoding="latin-1", header=None,
                      usecols=[0, 1], names=["label", "text"],
                      on_bad_lines="skip").dropna()
    spam = sms[sms.label == "spam"].sample(120, random_state=42)
    for i, text in enumerate(spam["text"].astype(str).map(clean)):
        rows.append({"message_id": f"s{i:03d}", "text": text,
                     **weak_labels(text), "notes": "draft: sms spam — verify labels"})

    # Ham SMS = negative examples (all-zero labels unless review says otherwise)
    ham = sms[sms.label == "ham"].sample(80, random_state=42)
    zero = {c: 0 for c in COLS}
    for i, text in enumerate(ham["text"].astype(str).map(clean)):
        rows.append({"message_id": f"h{i:03d}", "text": text,
                     **zero, "notes": "draft: legit ham — confirm no_tactic"})

    out = pd.DataFrame(rows, columns=["message_id", "text"] + COLS + ["notes"])
    out.to_csv("data/labeled_draft.csv", index=False)

    print(f"[OK] Wrote data/labeled_draft.csv with {len(out)} rows")
    print("\nPre-filled label counts (heuristic, before your review):")
    for c in COLS:
        print(f"  {c:20s} {int(out[c].sum()):4d}")
    no_tactic = (out[COLS].sum(axis=1) == 0).sum()
    print(f"  {'no_tactic':20s} {int(no_tactic):4d}")
    print("\n[NEXT] Review/correct every row, then save as data/labeled.csv")


if __name__ == "__main__":
    main()
