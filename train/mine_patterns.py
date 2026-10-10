"""
mine_patterns.py — Expand data/patterns/scam_patterns.jsonl with concrete
pattern variants mined from the ScamBench training data (data/labeled.csv).

One representative example per scenario_category, using the final (scam)
user turn of the conversation as the example snippet.

Run: python train/mine_patterns.py
"""

import json
import re
from pathlib import Path

import pandas as pd

from prepare_scambench import CATEGORY_TO_TACTICS, TACTIC_COLUMNS

PATTERNS_FILE = Path("data/patterns/scam_patterns.jsonl")
LABELED_CSV = "data/labeled.csv"
MAX_NEW = 30          # hand-written corpus has 20; aim for ~50 total
MIN_ROWS = 3          # only mine categories with at least this many rows

# Human-readable pattern names per category
PATTERN_NAMES = {
    "advance-fee-fraud": "Advance-fee fraud — pay a fee to unlock a larger sum",
    "bank-phishing": "Bank-branded phishing with fake security alert",
    "credential-phishing": "Fake login/verification page harvesting credentials",
    "credential-theft": "Direct request for passwords, OTPs, or PINs",
    "crypto": "Crypto escrow/doubling scheme",
    "crypto-investment-scam": "Guaranteed-return crypto investment platform",
    "dsi-impersonation": "Impersonation of a national investigation agency",
    "extortion": "Extortion — threats unless payment is made",
    "fake-billing": "Fake invoice or renewal billing notice",
    "fake-delivery-notification": "Fake parcel delivery fee notification",
    "government-impersonation": "Government/tax agency impersonation",
    "impersonation": "Trusted-person or institution impersonation",
    "impersonation-fraud": "Executive/colleague impersonation requesting action",
    "interpersonal-abuse": "Manipulative pressure within a personal relationship",
    "investment-fraud": "Fraudulent investment offer with guaranteed returns",
    "investment-scam": "Too-good-to-be-true investment opportunity",
    "job-scam": "Fake job offer requiring upfront payment",
    "loan-fraud": "Fake loan approval requiring an advance fee",
    "malicious-tool": "Prompts victim to install a malicious tool",
    "malware-delivery": "Delivers malware via attachment or download link",
    "payment-fraud": "Redirects a legitimate payment to a new account",
    "phishing": "Generic phishing with deceptive link",
    "phishing-link": "Phishing link disguised as an official portal",
    "promptpay-fraud": "Instant-payment (PromptPay) transfer fraud",
    "romance-scam": "Romance scam building trust before asking for money",
    "secret-exfiltration": "Asks the victim to reveal secrets or sensitive data",
    "shopping-fraud": "Fake online shop that never delivers",
    "smishing": "SMS phishing with urgent call-to-action",
    "social-engineering": "Authority + urgency social-engineering pressure",
    "social-manipulation": "Emotional manipulation to lower verification",
    "tech-support-scam": "Fake tech-support infection warning with call number",
    "unauthorized-access": "Seeks remote access to the victim's device/account",
    "voice-phishing": "Voice call phishing impersonating officials",
}

TACTIC_SHORT = {
    "t1_urgency": "urgency",
    "t2_authority": "authority",
    "t3_financial": "financial_lure",
    "t4_credential": "credential",
    "t5_payment": "payment_redirection",
    "t6_secrecy": "secrecy",
    "t7_tech_support": "tech_support",
}


def last_user_turn(text: str) -> str:
    """Extract the final user message from a flattened conversation."""
    turns = re.findall(r"^user:\s*(.+?)(?=^user:|^assistant:|\Z)",
                       text, flags=re.MULTILINE | re.DOTALL)
    snippet = turns[-1] if turns else text
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet[:240]


def main():
    existing = [json.loads(l) for l in
                PATTERNS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    max_id = max(int(p["id"].lstrip("p")) for p in existing)
    existing_names = {p["pattern"] for p in existing}

    df = pd.read_csv(LABELED_CSV)
    df = df[df["should_trigger"].astype(bool)]

    new_entries = []
    for category, name in PATTERN_NAMES.items():
        if len(new_entries) >= MAX_NEW:
            break
        if name in existing_names:
            continue  # already mined on a previous run
        subset = df[df["scenario_category"].str.lower() == category]
        if len(subset) < MIN_ROWS:
            continue

        # Pick a punchy representative example (fallback: any non-empty turn)
        candidates = subset["text"].map(last_user_turn)
        good = candidates[candidates.str.len().between(60, 400)]
        pool = good if not good.empty else candidates[candidates.str.len() > 0]
        if pool.empty:
            continue
        example = pool.loc[pool.str.len().idxmin()]

        row = subset.iloc[0]
        tactics = [TACTIC_SHORT[c] for c in TACTIC_COLUMNS if row[c] == 1]

        max_id += 1
        new_entries.append({
            "id": f"p{max_id:03d}",
            "tactic": "+".join(tactics),
            "pattern": name,
            "example": example,
            "source": f"ScamBench ({category})",
        })

    with PATTERNS_FILE.open("a", encoding="utf-8") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[OK] Added {len(new_entries)} mined patterns "
          f"(total {len(existing) + len(new_entries)})")
    for e in new_entries:
        print(f"  {e['id']}  [{e['tactic']}] {e['pattern']}")


if __name__ == "__main__":
    main()
