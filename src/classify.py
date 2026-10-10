"""
classify.py — Tactic classification inference.

Loads the fine-tuned DistilBERT model and returns per-tactic probabilities.
Multi-label: multiple tactics can be active for a single message.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = "models/tactic-classifier"

# Human-readable descriptions for the explanation layer
TACTIC_DESCRIPTIONS = {
    "urgency": "Creates artificial time pressure to prevent you from verifying",
    "authority": "Impersonates a trusted institution (bank, government, company)",
    "financial_lure": "Promises unexpected money, refunds, or prizes",
    "credential_harvesting": "Tries to get you to enter passwords, OTPs, or card details",
    "payment_redirection": "Asks you to send money via an untraceable method",
    "secrecy": "Tells you to keep it secret or avoid contacting officials",
    "tech_support": "Claims your device is infected and asks for remote access",
}


@dataclass
class TacticPrediction:
    """Structured tactic classification output."""
    tactics: List[str] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    top_tactic: Optional[str] = None
    top_probability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tactics": self.tactics,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "top_tactic": self.top_tactic,
            "top_probability": round(self.top_probability, 4),
        }

    def descriptions(self) -> List[str]:
        """Plain-language descriptions of the detected tactics."""
        return [TACTIC_DESCRIPTIONS[t] for t in self.tactics if t in TACTIC_DESCRIPTIONS]


class TacticClassifier:
    """Wrapper around the fine-tuned multi-label classifier."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: Optional[str] = None):
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}. "
                f"Train it first: python train/train_tactic_classifier.py --data data/labeled.csv"
            )

        with open(model_path / "label_metadata.json") as f:
            self.metadata = json.load(f)

        self.tactic_names = self.metadata["tactic_names"]
        self.threshold = self.metadata.get("threshold", 0.5)

        # Per-tactic tuned thresholds (train/tune_thresholds.py). Falls back
        # to the single metadata threshold for tactics missing from the file.
        thresholds_path = model_path / "thresholds.json"
        if thresholds_path.exists():
            raw = json.loads(thresholds_path.read_text(encoding="utf-8"))
            self.thresholds = {
                name: (float(v["threshold"]) if isinstance(v, dict) else float(v))
                for name, v in raw.items()
            }
        else:
            self.thresholds = {name: self.threshold for name in self.tactic_names}

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str, threshold: Optional[float] = None) -> TacticPrediction:
        """Classify a single message.

        By default each tactic fires at its own tuned threshold
        (thresholds.json). Pass `threshold` to override all tactics with a
        single value.
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        probabilities = {
            name: float(probs[i]) for i, name in enumerate(self.tactic_names)
        }
        active = [
            name for name, p in probabilities.items()
            if p >= (threshold if threshold is not None
                     else self.thresholds.get(name, 0.5))
        ]
        active.sort(key=lambda n: probabilities[n], reverse=True)

        top_tactic = max(probabilities, key=probabilities.get) if probabilities else None
        top_probability = probabilities.get(top_tactic, 0.0) if top_tactic else 0.0

        return TacticPrediction(
            tactics=active,
            probabilities=probabilities,
            top_tactic=top_tactic,
            top_probability=top_probability,
        )


# Singleton loader — avoids reloading the model on every Streamlit rerun
_classifier_instance: Optional[TacticClassifier] = None


def get_classifier() -> TacticClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = TacticClassifier()
    return _classifier_instance


# --- Quick test ---
if __name__ == "__main__":
    clf = get_classifier()
    test_msgs = [
        "URGENT: Your SBI account is suspended. Verify immediately at "
        "http://sbi-secure.xyz or call +91-9876543210 within 24 hours.",
        "Hi, your Amazon order #12345 has shipped. Track at amazon.com/orders.",
    ]
    for msg in test_msgs:
        pred = clf.predict(msg)
        print(f"\nMessage: {msg[:80]}...")
        print(f"  Tactics: {pred.tactics}")
        print(f"  Top: {pred.top_tactic} ({pred.top_probability:.3f})")
        for name, p in sorted(pred.probabilities.items(), key=lambda x: -x[1]):
            print(f"    {name:25s} {p:.3f}")
