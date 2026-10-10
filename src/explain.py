"""
explain.py — Grounded explanation generation via Ollama (qwen3 4B).

The LLM receives BOTH the original message AND the structured evidence.
It is instructed to describe only what is present in the message — never
to describe retrieved patterns as if they were the message itself.
"""

import json
from dataclasses import dataclass, field

import ollama

# Project-standard local model (see download_models.py). Any qwen3 4B-class
# tag works; resolve_model() falls back to whatever qwen3 variant is pulled.
MODEL = "qwen3:4b-instruct-2507-q8_0"

USER_PROMPT = """You are a scam-awareness assistant. You will be given a
message and structured evidence about it.

Rules for your explanation:
- Describe only what is ACTUALLY in the message. Do not describe retrieved
  patterns as if they were this message.
- If the evidence does not match the message, say the message looks fine.
- Do not invent links, fees, tactics, or details that are not present.
- Write at a 6th-grade reading level.
- If tactics are detected, name them.
- End with one concrete action step, or say "nothing to worry about" if the
  message looks legitimate.
- Keep it under 100 words.

MESSAGE:
\"\"\"
{message}
\"\"\"

STRUCTURED EVIDENCE:
{evidence}

Write the explanation now."""

BRIEF_PROMPT = """You are helping someone ask a trusted contact for a second
opinion on a suspicious message. Write a short, warm, non-alarming message
they can send to a family member or friend.

Rules:
- Only use what is actually in the message. Do not invent details.
- Do not include any personal information from the original message.
- Ask the contact to weigh in before the sender acts.
- Keep it under 80 words.

MESSAGE:
\"\"\"
{message}
\"\"\"

STRUCTURED EVIDENCE:
{evidence}

Write the message now."""


@dataclass
class Explanation:
    user_analysis: str = ""
    trusted_contact_brief: str = ""
    model: str = MODEL
    evidence_used: dict = field(default_factory=dict)


def resolve_model(preferred: str = MODEL) -> str:
    """Return `preferred` if pulled, else any installed qwen3 tag, else `preferred`.

    Keeps the demo working even if the exact tag differs from what's installed.
    """
    try:
        installed = [m.model for m in ollama.list().models]
        if preferred in installed:
            return preferred
        for name in installed:
            if name.startswith("qwen3"):
                return name
    except Exception:
        pass
    return preferred


def _format_evidence(evidence: dict) -> str:
    return json.dumps(evidence, indent=2, ensure_ascii=False)


# --- Deterministic fallback (used when Ollama is unavailable) ---
_TACTIC_PHRASES = {
    "urgency": "artificial time pressure",
    "authority": "someone claiming official authority",
    "financial_lure": "a promise of unexpected money",
    "credential_harvesting": "an attempt to collect passwords or codes",
    "payment_redirection": "a request to send money an unusual way",
    "secrecy": "a demand to keep this secret",
    "tech_support": "a claim that your device is infected",
}


def _detected_tactics(evidence: dict) -> list:
    tactics = evidence.get("tactics")
    if not tactics:
        probs = evidence.get("tactic_probabilities") or {}
        thresholds = evidence.get("tactic_thresholds") or {}
        tactics = [t for t, p in probs.items() if p >= thresholds.get(t, 0.5)]
    return tactics


def _template_user_analysis(evidence: dict) -> str:
    tactics = _detected_tactics(evidence)
    url_flags = evidence.get("url_flags") or []
    parts = []
    if tactics:
        named = ", ".join(_TACTIC_PHRASES.get(t, t) for t in tactics)
        parts.append(f"This message shows common scam signals: {named}.")
    if url_flags:
        parts.append(f"The link is suspicious: {url_flags[0]}.")
    if not tactics and not url_flags:
        parts.append("No strong scam signals were detected in this message.")
        parts.append("If it still feels wrong, trust that feeling and verify before acting.")
    else:
        parts.append("Do not reply, click any link, or share codes. If it claims to be "
                     "from a bank or company, contact them using the number on your card "
                     "or their official app — never the contact details in this message.")
    return " ".join(parts)


def _template_brief(evidence: dict) -> str:
    tactics = _detected_tactics(evidence)
    if tactics:
        named = ", ".join(_TACTIC_PHRASES.get(t, t) for t in tactics[:2])
        hint = f"a scam — it uses {named}"
    else:
        hint = "something that looks off"
    return (
        "Hi — I got a message I'm not sure about. It looks like "
        f"{hint}. Could you take a quick look before I do anything? "
        "I'd rather double-check with you first."
    )


def _call_ollama(prompt: str, model: str) -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 300},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def generate_explanation(evidence: dict, message: str = "") -> Explanation:
    """
    Generate a grounded explanation.

    `message` is the original text — required for grounding. If omitted,
    the model will still run but may describe patterns as if they were
    the message (this was the bug we fixed on Oct 8).

    If Ollama is unavailable (e.g. HuggingFace Spaces, no GPU), falls back
    to a deterministic template built strictly from the evidence — the
    demo never hard-fails and never invents details either way.
    """
    model = resolve_model()
    formatted = _format_evidence(evidence)

    user_analysis = _call_ollama(
        USER_PROMPT.format(message=message, evidence=formatted), model)
    if not user_analysis or user_analysis.startswith("[LLM unavailable"):
        user_analysis = _template_user_analysis(evidence)

    brief = _call_ollama(
        BRIEF_PROMPT.format(message=message, evidence=formatted), model)
    if not brief or brief.startswith("[LLM unavailable"):
        brief = _template_brief(evidence)

    return Explanation(
        user_analysis=user_analysis,
        trusted_contact_brief=brief,
        model=model,
        evidence_used=evidence,
    )


if __name__ == "__main__":
    sample_msg = "Hi, this is Dr. Mehta's office confirming your appointment on Thursday at 3 PM."
    sample_ev = {"tactics": [], "risk_score": 12, "retrieved_patterns": []}
    exp = generate_explanation(sample_ev, message=sample_msg)
    print("=== USER ANALYSIS ===\n", exp.user_analysis)
    print("\n=== BRIEF ===\n", exp.trusted_contact_brief)
