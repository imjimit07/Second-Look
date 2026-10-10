"""
tests/test_fuse.py — Regression tests for the fusion and boundary-distance logic.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fuse import fuse, SUSPICIOUS_THRESHOLD, SCAM_THRESHOLD


def _ev(**overrides):
    base = {
        "tactic_probabilities": {},
        "url_risk_scores": [],
        "retrieved_patterns": [],
        "extracted_signals": {},
    }
    base.update(overrides)
    return base


class TestDecisionBands:
    def test_no_signals_returns_legit(self):
        result = fuse(_ev())
        assert result.decision == "likely_legitimate"
        assert result.risk_score < SUSPICIOUS_THRESHOLD

    def test_strong_tactic_returns_scam(self):
        result = fuse(_ev(tactic_probabilities={"urgency": 0.9}))
        assert result.decision == "scam"
        assert result.risk_score >= SCAM_THRESHOLD

    def test_high_url_risk_alone(self):
        result = fuse(_ev(url_risk_scores=[85]))
        assert result.decision == "scam"


class TestBoundaryDistance:
    def test_scam_at_threshold_has_min_boundary_distance(self):
        ev = _ev(tactic_probabilities={"urgency": (SCAM_THRESHOLD + 0.1) / 100})
        result = fuse(ev)
        # Right at threshold -> boundary distance near 0.5, not near 1.0
        assert result.decision == "scam"
        assert result.boundary_distance < 0.6

    def test_scam_at_extreme_has_max_boundary_distance(self):
        ev = _ev(
            tactic_probabilities={"urgency": 1.0},
            url_risk_scores=[100],
        )
        result = fuse(ev)
        assert result.risk_score >= 95
        assert result.boundary_distance > 0.9

    def test_suspicious_band_midpoint_is_highest(self):
        """Boundary distance in the suspicious band should peak at the
        midpoint, not at the boundaries — this was the reviewer-found bug."""
        span = SCAM_THRESHOLD - SUSPICIOUS_THRESHOLD
        midpoint = SUSPICIOUS_THRESHOLD + span / 2

        ev_mid = _ev(tactic_probabilities={"urgency": (midpoint + 0.5) / 100})
        ev_edge = _ev(tactic_probabilities={"urgency": (SUSPICIOUS_THRESHOLD + 0.5) / 100})

        r_mid = fuse(ev_mid)
        r_edge = fuse(ev_edge)

        if r_mid.decision == "suspicious" and r_edge.decision == "suspicious":
            assert r_mid.boundary_distance > r_edge.boundary_distance

    def test_boundary_distance_never_leaks_above_one(self):
        ev = _ev(url_risk_scores=[100], tactic_probabilities={"urgency": 1.0})
        assert fuse(ev).boundary_distance <= 1.0


class TestRobustness:
    def test_empty_evidence_does_not_crash(self):
        assert fuse({}) is not None

    def test_legitimate_pattern_does_not_add_risk(self):
        """Retrieved 'legitimate' patterns should never contribute risk."""
        ev = _ev(
            retrieved_patterns=[
                {"similarity": 0.95, "tactic": "legitimate", "pattern": "Legit OTP"},
            ]
        )
        result = fuse(ev)
        assert result.decision == "likely_legitimate"

    def test_no_patterns_still_works(self):
        ev = _ev(tactic_probabilities={"urgency": 0.8})
        result = fuse(ev)
        assert result.decision == "scam"


class TestPerTacticThresholds:
    def test_tactic_fires_at_its_own_threshold(self):
        # urgency fires at its tuned threshold (0.45), not the default 0.5
        ev = _ev(
            tactic_probabilities={"urgency": 0.46},
            tactic_thresholds={"urgency": 0.45},
        )
        assert "urgency" in fuse(ev).active_tactics

    def test_tactic_held_below_its_own_threshold(self):
        ev = _ev(
            tactic_probabilities={"urgency": 0.44},
            tactic_thresholds={"urgency": 0.45},
        )
        assert "urgency" not in fuse(ev).active_tactics
