"""
fuse.py — Unified evidence fusion and decision.

Takes structured evidence from classifier, URL features, retrieval, and
keyword extraction. Returns a single FusionResult used by BOTH the
Streamlit app and the eval harness, so eval metrics describe exactly
what the app does.

Design:
- Tactic classifier and URL features are PRIMARY signals, combined by max
  (they're orthogonal — a fresh phishing domain may score low on tactics
  but high on URL features, and vice versa).
- Co-occurring tactics and retrieved pattern similarity are CORROBORATING
  signals added on top, capped so they can't alone push a weak primary
  over the line.
- Raw keyword counts are the weakest signal — they exist to break ties,
  not to decide.
- Decision bands are on the fused 0-100 score, not on any single component.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Tunable thresholds (winner of eval/grid_fusion.py on 600-row balanced val) ---
SCAM_THRESHOLD = 55
SUSPICIOUS_THRESHOLD = 40

# Component weights (only used inside the fusion formula)
CO_OCCURRENCE_BONUS_PER_TACTIC = 3   # each additional active tactic
CO_OCCURRENCE_BONUS_CAP = 15
CORROBORATION_CAP = 15
KEYWORD_WEIGHT = 0.15
PATTERN_WEIGHT = 0.15
PATTERN_PRIMARY_SCALE = 0.8          # winner of grid search; top-10 combos were within ~1 row


@dataclass
class FusionResult:
    decision: str                       # "scam" | "suspicious" | "likely_legitimate"
    risk_score: int                     # 0-100
    boundary_distance: float            # 0.5-1.0, how far the score is from the
                                        # nearest decision boundary — NOT a
                                        # calibrated probability
    active_tactics: List[str] = field(default_factory=list)
    tactic_probabilities: Dict[str, float] = field(default_factory=dict)
    contributing_factors: List[str] = field(default_factory=list)
    evidence_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "risk_score": self.risk_score,
            "boundary_distance": round(self.boundary_distance, 3),
            "active_tactics": self.active_tactics,
            "tactic_probabilities": {k: round(v, 4) for k, v in self.tactic_probabilities.items()},
            "contributing_factors": self.contributing_factors,
            "evidence_breakdown": self.evidence_breakdown,
        }

    @property
    def is_scam(self) -> bool:
        return self.decision == "scam"


def fuse(evidence: Dict[str, Any]) -> FusionResult:
    """
    Fuse evidence into a decision.

    Expected `evidence` keys (all optional — missing ones are treated as empty):
        tactic_probabilities: Dict[str, float]   (0-1 each)
        tactic_thresholds:     Dict[str, float]   (per-tactic fire thresholds; default 0.5)
        url_risk_scores:      List[int]          (0-100 each)
        retrieved_patterns:   List[dict]         (each has 'similarity' 0-1; entries with
                                                  tactic=='legitimate' are ignored)
        extracted_signals:    Dict[str, List]    (urgency/financial/authority lists)
    """
    tactic_probs: Dict[str, float] = evidence.get("tactic_probabilities", {}) or {}
    tactic_thresholds: Dict[str, float] = evidence.get("tactic_thresholds", {}) or {}
    url_risk_scores: List[int] = evidence.get("url_risk_scores", []) or []
    patterns: List[dict] = evidence.get("retrieved_patterns", []) or []
    signals: Dict[str, List] = evidence.get("extracted_signals", {}) or {}

    # Legit-pattern matches are reassurance, not scam evidence — never let
    # them contribute to the risk score.
    patterns = [p for p in patterns if p.get("tactic") != "legitimate"]

    # --- Component 1: tactic classifier (primary) ---
    top_prob = max(tactic_probs.values()) if tactic_probs else 0.0
    active_tactics = sorted(
        [t for t, p in tactic_probs.items()
         if p >= tactic_thresholds.get(t, 0.5)],
        key=lambda t: tactic_probs[t],
        reverse=True,
    )
    extra_tactics = max(len(active_tactics) - 1, 0)
    co_occurrence = min(
        extra_tactics * CO_OCCURRENCE_BONUS_PER_TACTIC,
        CO_OCCURRENCE_BONUS_CAP,
    )
    tactic_primary = min(top_prob * 100 + co_occurrence, 100)

    # --- Component 2: URL risk (primary) ---
    url_primary = max(url_risk_scores) if url_risk_scores else 0

    # --- Component 3: retrieval similarity (semi-primary, scaled down) ---
    top_similarity = patterns[0]["similarity"] if patterns else 0.0
    pattern_primary = top_similarity * 100 * PATTERN_PRIMARY_SCALE

    # Base = the strongest primary signal
    base = max(tactic_primary, url_primary, pattern_primary)

    # --- Corroboration: keyword signals + pattern similarity ---
    keyword_count = sum(len(v) for v in signals.values())
    keyword_corroboration = min(keyword_count * KEYWORD_WEIGHT * 5, CORROBORATION_CAP / 2)
    pattern_corroboration = min(top_similarity * 100 * PATTERN_WEIGHT, CORROBORATION_CAP / 2)
    corroboration = min(keyword_corroboration + pattern_corroboration, CORROBORATION_CAP)

    risk_score = int(min(base + corroboration, 100))

    # --- Decision band ---
    if risk_score >= SCAM_THRESHOLD:
        decision = "scam"
    elif risk_score >= SUSPICIOUS_THRESHOLD:
        decision = "suspicious"
    else:
        decision = "likely_legitimate"

    # Boundary distance = how far the score is from the nearest decision
    # boundary, normalized to [0.5, 1.0]. At a boundary: 0.5 (could flip
    # either way). At the extreme: 1.0. This is a distance measure, not a
    # calibrated probability — see README.
    if decision == "scam":
        span = 100 - SCAM_THRESHOLD
        boundary_distance = 0.5 + 0.5 * min((risk_score - SCAM_THRESHOLD) / span, 1.0)
    elif decision == "suspicious":
        span = SCAM_THRESHOLD - SUSPICIOUS_THRESHOLD
        distance_from_edge = min(
            risk_score - SUSPICIOUS_THRESHOLD,
            SCAM_THRESHOLD - risk_score,
        )
        boundary_distance = 0.5 + 0.5 * (distance_from_edge / (span / 2))
    else:  # likely_legitimate
        boundary_distance = 0.5 + 0.5 * min(
            (SUSPICIOUS_THRESHOLD - risk_score) / SUSPICIOUS_THRESHOLD, 1.0
        )

    # --- Human-readable contributing factors (for explanation + video) ---
    factors: List[str] = []
    if tactic_primary >= 50:
        factors.append(f"Tactic classifier: {active_tactics[0] if active_tactics else 'unknown'} ({top_prob:.2f})")
    if len(active_tactics) >= 2:
        factors.append(f"Multiple tactics co-occurring: {', '.join(active_tactics[:3])}")
    if url_primary >= 40:
        factors.append(f"URL risk features: {url_primary}/100")
    if top_similarity >= 0.5:
        factors.append(f"Matches known scam pattern (similarity {top_similarity:.2f})")
    if keyword_count > 0:
        factors.append(f"{keyword_count} keyword signal(s): {', '.join(k for k, v in signals.items() if v)}")

    return FusionResult(
        decision=decision,
        risk_score=risk_score,
        boundary_distance=float(boundary_distance),
        active_tactics=active_tactics,
        tactic_probabilities=tactic_probs,
        contributing_factors=factors,
        evidence_breakdown={
            "tactic_primary": round(tactic_primary, 1),
            "url_primary": url_primary,
            "pattern_primary": round(pattern_primary, 1),
            "base": round(base, 1),
            "corroboration": round(corroboration, 1),
            "co_occurrence_bonus": co_occurrence,
        },
    )
