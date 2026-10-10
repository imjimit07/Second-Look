"""
app.py — Second Look Streamlit frontend.

Polished UI: hides Streamlit chrome, color-coded verdict, custom styling.
Same pipeline as the eval harness — the fusion logic in src/fuse.py is
shared, so what the eval measures is what the app does.
"""

import streamlit as st

from src.classify import get_classifier
from src.explain import generate_explanation
from src.extract import extract_all
from src.features_url import analyze_urls
from src.fuse import fuse
from src.guidance import build_guidance
from src.retrieve import get_retriever

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Second Look",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Custom CSS — hides Streamlit chrome, applies dark theme, colors the verdict
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Hide Streamlit's default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stStatusWidget"] {visibility: hidden;}

    /* Page background + typography */
    .stApp { background-color: #0a0a0b; }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 4rem;
        max-width: 780px;
    }
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        color: #e8e8ed;
    }

    /* Logo header */
    .sl-header {
        display: flex; align-items: center; gap: 10px;
        font-size: 22px; font-weight: 600;
        letter-spacing: -0.02em; margin-bottom: 6px;
    }
    .sl-mark {
        width: 30px; height: 30px; border-radius: 8px;
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        display: grid; place-items: center; font-size: 16px;
    }
    .sl-tagline {
        color: #8b8b94; font-size: 15px;
        line-height: 1.55; margin-bottom: 28px;
    }

    /* Textarea */
    .stTextArea textarea {
        background-color: #131317 !important;
        border: 1px solid #23232a !important;
        border-radius: 14px !important;
        color: #e8e8ed !important;
        font-family: system-ui, -apple-system, sans-serif !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 16px !important;
    }
    .stTextArea textarea:focus {
        border-color: #32323c !important;
        box-shadow: none !important;
    }
    .stTextArea textarea::placeholder { color: #5a5a63 !important; }

    /* Primary button */
    .stButton > button {
        background-color: #f59e0b !important;
        color: #1a1a1a !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background-color: #fbbf24 !important;
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }

    /* Verdict card */
    .sl-verdict {
        border-radius: 14px; padding: 22px 24px;
        margin-top: 8px; margin-bottom: 24px;
        border: 1px solid #23232a;
        background: #131317;
    }
    .sl-verdict.scam {
        border-color: rgba(239, 68, 68, 0.35);
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.07), rgba(239, 68, 68, 0.01));
    }
    .sl-verdict.suspicious {
        border-color: rgba(245, 158, 11, 0.35);
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.07), rgba(245, 158, 11, 0.01));
    }
    .sl-verdict.likely_legitimate {
        border-color: rgba(16, 185, 129, 0.35);
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.07), rgba(16, 185, 129, 0.01));
    }
    .sl-verdict-label {
        display: flex; align-items: center; gap: 8px;
        font-size: 12px; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        margin-bottom: 10px;
    }
    .sl-dot {
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block;
    }
    .scam .sl-dot { background: #ef4444; box-shadow: 0 0 12px #ef4444; }
    .suspicious .sl-dot { background: #f59e0b; box-shadow: 0 0 12px #f59e0b; }
    .likely_legitimate .sl-dot { background: #10b981; box-shadow: 0 0 12px #10b981; }
    .scam .sl-verdict-label { color: #ef4444; }
    .suspicious .sl-verdict-label { color: #f59e0b; }
    .likely_legitimate .sl-verdict-label { color: #10b981; }
    .sl-verdict-title {
        font-size: 24px; font-weight: 600;
        letter-spacing: -0.02em; line-height: 1.25;
        color: #e8e8ed;
    }
    .sl-verdict-meta {
        display: flex; gap: 22px; margin-top: 14px;
        font-family: ui-monospace, monospace;
        font-size: 13px; color: #8b8b94;
    }
    .sl-verdict-meta strong { color: #e8e8ed; font-weight: 500; }

    /* Section labels */
    .sl-section {
        font-size: 12px; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #5a5a63; margin-top: 28px; margin-bottom: 12px;
    }

    /* Tactic tag */
    .sl-tag {
        display: inline-block;
        background: #131317;
        border: 1px solid #32323c;
        border-radius: 8px;
        padding: 7px 12px;
        font-size: 13px;
        margin: 0 6px 6px 0;
    }
    .sl-tag-prob {
        font-family: ui-monospace, monospace;
        font-size: 12px; color: #f59e0b;
        margin-left: 6px;
    }

    /* Evidence item */
    .sl-evidence {
        font-family: ui-monospace, monospace;
        font-size: 13px;
        color: #8b8b94;
        padding: 8px 12px;
        background: #131317;
        border-radius: 6px;
        border-left: 2px solid #32323c;
        margin-bottom: 6px;
    }
    .sl-evidence.hot {
        border-left-color: #f59e0b;
        color: #e8e8ed;
    }

    /* Prose box (explanation) */
    .sl-prose {
        background: #131317;
        border: 1px solid #23232a;
        border-radius: 12px;
        padding: 18px;
        color: #e8e8ed;
        font-size: 15px;
        line-height: 1.65;
    }

    /* Steps list */
    .sl-step {
        display: flex; gap: 14px;
        background: #131317;
        border: 1px solid #23232a;
        border-radius: 12px;
        padding: 15px 16px;
        margin-bottom: 8px;
    }
    .sl-step-num {
        width: 26px; height: 26px;
        border-radius: 50%;
        display: grid; place-items: center;
        font-family: ui-monospace, monospace;
        font-size: 12px; font-weight: 600;
        flex-shrink: 0; margin-top: 1px;
        background: #32323c; color: #e8e8ed;
    }
    .sl-step.immediate .sl-step-num { background: #ef4444; color: #ffffff; }
    .sl-step.verify .sl-step-num { background: #f59e0b; color: #1a1a1a; }
    .sl-step.report .sl-step-num { background: #5a5a63; color: #e8e8ed; }
    .sl-step-title { font-weight: 600; margin-bottom: 3px; color: #e8e8ed; font-size: 15px; }
    .sl-step-detail { color: #8b8b94; font-size: 14px; line-height: 1.55; }

    /* Brief card (the trusted-contact message) */
    .sl-brief {
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.09), rgba(245, 158, 11, 0.02));
        border: 1px solid rgba(245, 158, 11, 0.32);
        border-radius: 14px;
        padding: 22px;
        margin-top: 4px;
    }
    .sl-brief-label {
        color: #f59e0b;
        font-size: 12px; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        margin-bottom: 12px;
    }
    .sl-brief-text {
        color: #e8e8ed;
        font-size: 15px; line-height: 1.65;
        white-space: pre-wrap;
    }

    /* Expander (escalation) */
    .streamlit-expanderHeader {
        background-color: #131317 !important;
        border-radius: 10px !important;
        color: #8b8b94 !important;
        font-size: 14px !important;
    }
    .streamlit-expanderContent {
        background-color: #131317 !important;
        border: 1px solid #23232a !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 16px !important;
    }
    details[open] summary { color: #e8e8ed !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: #f59e0b !important; }

    /* Warning / error text */
    .stAlert { background-color: #131317 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="sl-header">
    <div class="sl-mark">🔍</div>
    <span>Second Look</span>
</div>
<div class="sl-tagline">
    Paste a suspicious message. Get the tactic, the evidence, and what to do next —
    plus a note you can send to someone you trust.
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
message = st.text_area(
    "Message",
    height=160,
    placeholder="Paste the suspicious SMS, email, or DM here…",
    label_visibility="collapsed",
)

col_a, col_b = st.columns([3, 1])
with col_b:
    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
with col_a:
    st.markdown(
        '<div style="color:#5a5a63;font-size:13px;padding-top:10px;">'
        'Runs entirely on your machine — nothing is sent anywhere.'
        '</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
VERDICT_LABELS = {
    "scam": "Likely a scam",
    "suspicious": "Suspicious — verify first",
    "likely_legitimate": "No clear scam signals",
}
VERDICT_TITLES = {
    "scam": "Do not reply, click, or pay.",
    "suspicious": "Don't act until you've verified.",
    "likely_legitimate": "Nothing clearly fraudulent here.",
}


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


if analyze_clicked and message.strip():
    with st.spinner("Analyzing locally…"):
        # --- Same pipeline as eval ---
        entities = extract_all(message)
        tactic_pred = get_classifier().predict(message)
        url_features = analyze_urls(entities.urls) if entities.urls else []
        retrieval = get_retriever().search(message, top_k=3, min_similarity=0.6)

        evidence = {
            "tactics": tactic_pred.tactics,
            "probabilities": tactic_pred.probabilities,
            "tactic_probabilities": tactic_pred.probabilities,
            "tactic_thresholds": get_classifier().thresholds,   # tuned per tactic
            "url_flags": [d for f in url_features for d in f.risk_details],
            "url_risk_scores": [f.risk_score for f in url_features],
            "retrieved_patterns": [m.to_dict() for m in retrieval.matches],
            "extracted_signals": {
                "urgency": entities.urgency_signals,
                "financial": entities.financial_signals,
                "authority": entities.authority_signals,
            },
        }

        fusion = fuse(evidence)
        guidance = build_guidance(fusion)
        explanation = generate_explanation(evidence, message=message)

    # --- Verdict card ---
    st.markdown(f"""
    <div class="sl-verdict {fusion.decision}">
        <div class="sl-verdict-label">
            <span class="sl-dot"></span>
            <span>{VERDICT_LABELS[fusion.decision]}</span>
        </div>
        <div class="sl-verdict-title">{VERDICT_TITLES[fusion.decision]}</div>
        <div class="sl-verdict-meta">
            <span>risk <strong>{fusion.risk_score}/100</strong></span>
            <span>confidence <strong>{int(fusion.confidence * 100)}%</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Detected tactics ---
    if fusion.active_tactics:
        st.markdown('<div class="sl-section">Detected tactics</div>', unsafe_allow_html=True)
        tags_html = ""
        for t in fusion.active_tactics:
            prob = fusion.tactic_probabilities.get(t, 0.0)
            label = t.replace("_", " ")
            tags_html += f'<span class="sl-tag">{label}<span class="sl-tag-prob">{prob:.2f}</span></span>'
        st.markdown(f'<div>{tags_html}</div>', unsafe_allow_html=True)

    # --- Evidence ---
    evidence_items = list(fusion.contributing_factors)
    for f in url_features:
        if f.risk_score > 0:
            evidence_items.extend([f"URL · {d}" for d in f.risk_details])
    for p in retrieval.matches[:2]:
        evidence_items.append(f"Pattern match ({p.similarity:.2f}) · {p.pattern}")

    if evidence_items:
        st.markdown('<div class="sl-section">Evidence</div>', unsafe_allow_html=True)
        ev_html = ""
        for item in evidence_items[:8]:
            hot = " hot" if ("Pattern" in item or "URL" in item or "classifier" in item) else ""
            ev_html += f'<div class="sl-evidence{hot}">{_esc(item)}</div>'
        st.markdown(ev_html, unsafe_allow_html=True)

    # --- Explanation ---
    st.markdown('<div class="sl-section">What this means</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sl-prose">{_esc(explanation.user_analysis)}</div>',
        unsafe_allow_html=True,
    )

    # --- Action steps ---
    if guidance.steps:
        st.markdown('<div class="sl-section">What to do</div>', unsafe_allow_html=True)
        steps_html = ""
        for i, s in enumerate(guidance.steps, 1):
            steps_html += f"""
            <div class="sl-step {s.priority}">
                <div class="sl-step-num">{i}</div>
                <div>
                    <div class="sl-step-title">{_esc(s.title)}</div>
                    <div class="sl-step-detail">{_esc(s.detail)}</div>
                </div>
            </div>
            """
        st.markdown(steps_html, unsafe_allow_html=True)

    # --- Trusted-contact brief (the novel bit) — hidden for clean verdicts ---
    if fusion.decision != "likely_legitimate":
        st.markdown('<div class="sl-section">Share with someone you trust</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="sl-brief">
                <div class="sl-brief-label">Message for a family member or friend</div>
                <div class="sl-brief-text">{_esc(explanation.trusted_contact_brief)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Streamlit's copy path — expander with a st.code block (built-in copy button)
        with st.expander("Copy this message as text", expanded=False):
            st.code(explanation.trusted_contact_brief, language=None)

    # --- Escalation (if already acted) ---
    if guidance.escalation_steps:
        with st.expander("If you already acted on this message"):
            for s in guidance.escalation_steps:
                st.markdown(f"**{s.title}**")
                st.markdown(
                    f'<div style="color:#8b8b94;font-size:14px;margin-bottom:14px;">{_esc(s.detail)}</div>',
                    unsafe_allow_html=True,
                )

elif analyze_clicked:
    st.warning("Paste a message first.")
