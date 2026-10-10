"""
guidance.py — Action guidance keyed to tactics and decision band.

Deterministic, template-based. No LLM. This is what makes the tool
*actionable* rather than just an explanation — and it can't fail on
recording day.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

if __package__ in (None, ""):  # allow running as: python src/guidance.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fuse import FusionResult


@dataclass
class ActionStep:
    priority: str    # "immediate" | "verify" | "report"
    title: str
    detail: str

    def to_dict(self) -> dict:
        return {"priority": self.priority, "title": self.title, "detail": self.detail}


@dataclass
class Guidance:
    decision: str
    summary: str
    steps: List[ActionStep] = field(default_factory=list)
    report_links: List[dict] = field(default_factory=list)
    escalation_steps: List[ActionStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "report_links": self.report_links,
            "escalation_steps": [s.to_dict() for s in self.escalation_steps],
        }


# --- Tactic-specific first actions ---
TACTIC_ACTIONS = {
    "urgency": ActionStep(
        "immediate",
        "Slow down",
        "Real institutions never require an answer in minutes. Anything with a "
        "countdown is designed to stop you from thinking. You have time.",
    ),
    "authority": ActionStep(
        "verify",
        "Verify through a channel you chose",
        "Do not use the number, link, or email in the message. Look up the "
        "organization yourself — the number on your bank card, the official app, "
        "or the website you'd type yourself.",
    ),
    "financial_lure": ActionStep(
        "immediate",
        "You did not win anything",
        "If you didn't enter a contest, you didn't win one. Refunds and prizes "
        "you weren't expecting are the scam, not the reward.",
    ),
    "credential_harvesting": ActionStep(
        "immediate",
        "Do not enter your password or OTP",
        "No legitimate bank or company will ask for your password, OTP, or full "
        "card number by message. If you've already entered them, change that "
        "password now on the official site.",
    ),
    "payment_redirection": ActionStep(
        "immediate",
        "Do not send money via gift card, crypto, or wire",
        "These payment methods are irreversible and untraceable. No legitimate "
        "business, government, or relative will ask you to pay this way. If a "
        "payment destination 'changed', call the recipient on a number you "
        "already had.",
    ),
    "secrecy": ActionStep(
        "immediate",
        "Tell someone you trust, right now",
        "Secrecy is the scammer's most important tool. Anyone telling you to keep "
        "something secret from family, friends, or your bank is isolating you on "
        "purpose. Say it out loud to another person before you act.",
    ),
    "tech_support": ActionStep(
        "immediate",
        "Microsoft, Apple, and Google will never call you",
        "If your device is actually infected, they would not notify you by "
        "message or phone. Do not install AnyDesk, TeamViewer, or any screen-"
        "sharing tool for a stranger.",
    ),
}

# --- Decision-band summaries ---
BAND_SUMMARIES = {
    "scam": "This looks like a scam. Do not reply, click, or pay. The steps below are in order — start at the top.",
    "suspicious": "Some parts of this look off. Don't act on it until you've verified through a channel you chose yourself.",
    "likely_legitimate": "Nothing in this message is clearly a scam signal. If it still feels wrong, trust that feeling and verify before acting.",
}

# --- Report links ---
REPORT_LINKS = [
    {"name": "FTC (US)", "url": "https://reportfraud.ftc.gov"},
    {"name": "FBI IC3 (US)", "url": "https://www.ic3.gov"},
    {"name": "Action Fraud (UK)", "url": "https://www.actionfraud.police.uk"},
    {"name": "Cyber Crime Portal (IN)", "url": "https://cybercrime.gov.in"},
]

# --- Escalation (if the user already acted) ---
ESCALATION_STEPS = [
    ActionStep(
        "immediate",
        "Contact your bank or card issuer now",
        "Call the number on the back of your card. Ask them to freeze the "
        "transaction or the account. The first hour matters.",
    ),
    ActionStep(
        "immediate",
        "Change passwords you may have entered",
        "Change the password on any account you may have typed credentials into, "
        "and enable two-factor authentication on it.",
    ),
    ActionStep(
        "report",
        "File a report",
        "Reporting helps authorities link cases. It does not guarantee recovery, "
        "but it is the only path that has a chance of one.",
    ),
]


def build_guidance(fusion: FusionResult) -> Guidance:
    """Build ordered action guidance from a FusionResult."""
    steps: List[ActionStep] = []

    # Lead with tactic-specific actions, ordered by priority
    priority_order = {"immediate": 0, "verify": 1, "report": 2}
    tactic_steps = [TACTIC_ACTIONS[t] for t in fusion.active_tactics if t in TACTIC_ACTIONS]
    tactic_steps.sort(key=lambda s: priority_order.get(s.priority, 9))
    steps.extend(tactic_steps)

    # For scam decisions with no tactic-specific steps, add a generic one
    if fusion.decision == "scam" and not tactic_steps:
        steps.append(ActionStep(
            "immediate",
            "Do not engage",
            "Do not reply, click any link, or call any number in this message.",
        ))

    # Always end with a verification step for non-legitimate decisions
    if fusion.decision in ("scam", "suspicious"):
        steps.append(ActionStep(
            "verify",
            "Ask someone you trust",
            "Show this message to a family member or friend before you do anything. "
            "A second opinion is the single most effective protection against scams.",
        ))

    guidance = Guidance(
        decision=fusion.decision,
        summary=BAND_SUMMARIES[fusion.decision],
        steps=steps,
        report_links=REPORT_LINKS if fusion.decision == "scam" else [],
    )

    # Escalation advice is always included but framed as "if you already acted"
    if fusion.decision == "scam":
        guidance.escalation_steps = list(ESCALATION_STEPS)

    return guidance


# --- Quick test ---
if __name__ == "__main__":
    from src.fuse import fuse
    sample_evidence = {
        "tactic_probabilities": {
            "urgency": 0.81, "authority": 0.72, "credential_harvesting": 0.68,
            "financial_lure": 0.12, "payment_redirection": 0.09,
            "secrecy": 0.11, "tech_support": 0.04,
        },
        "url_risk_scores": [45],
        "retrieved_patterns": [{"similarity": 0.78}],
        "extracted_signals": {"urgency": ["immediately", "suspended"], "authority": ["bank"]},
    }
    fusion = fuse(sample_evidence)
    print("FUSION:", fusion.to_dict())
    guidance = build_guidance(fusion)
    print("\nGUIDANCE SUMMARY:", guidance.summary)
    for s in guidance.steps:
        print(f"  [{s.priority}] {s.title}: {s.detail[:60]}...")
