"""
retrieve.py — Nearest-neighbor retrieval over the scam pattern corpus.

Given a suspicious message, returns the top-k most similar known scam
patterns. These become cited evidence in the explanation layer, grounding
the LLM and preventing hallucinated red flags.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("models/pattern_index")


@dataclass
class RetrievedPattern:
    id: str
    pattern: str
    example: str
    tactic: str
    source: str
    similarity: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "example": self.example,
            "tactic": self.tactic,
            "source": self.source,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class RetrievalResult:
    matches: List[RetrievedPattern] = field(default_factory=list)
    top_similarity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "matches": [m.to_dict() for m in self.matches],
            "top_similarity": round(self.top_similarity, 4),
        }


class PatternRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        if not index_dir.exists():
            raise FileNotFoundError(
                f"Pattern index not found at {index_dir}. "
                f"Build it: python train/build_pattern_index.py"
            )
        self.index = faiss.read_index(str(index_dir / "patterns.faiss"))
        with open(index_dir / "patterns.json", encoding="utf-8") as f:
            self.patterns = json.load(f)
        with open(index_dir / "config.json", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model = SentenceTransformer(self.config["embed_model"])

    def search(self, text: str, top_k: int = 3, min_similarity: float = 0.6) -> RetrievalResult:
        """Return top-k patterns above the similarity floor."""
        emb = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(emb, top_k)

        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_similarity:
                continue
            p = self.patterns[idx]
            matches.append(
                RetrievedPattern(
                    id=p["id"],
                    pattern=p["pattern"],
                    example=p["example"],
                    tactic=p.get("tactic", ""),
                    source=p.get("source", ""),
                    similarity=float(score),
                )
            )

        top_sim = matches[0].similarity if matches else 0.0
        return RetrievalResult(matches=matches, top_similarity=top_sim)


_retriever_instance: Optional[PatternRetriever] = None


def get_retriever() -> PatternRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = PatternRetriever()
    return _retriever_instance


# --- Quick test ---
if __name__ == "__main__":
    r = get_retriever()
    query = "Your account has been suspended. Verify immediately at the link below."
    result = r.search(query, top_k=3)
    print(f"Query: {query}")
    print(f"Top similarity: {result.top_similarity:.3f}\n")
    for m in result.matches:
        print(f"  [{m.similarity:.3f}] {m.pattern}")
        print(f"           tactic={m.tactic}")
        print(f"           source={m.source}\n")
