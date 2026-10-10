"""
build_pattern_index.py — Build a FAISS index over the scam pattern corpus.

Embeds each pattern using sentence-transformers and saves the index
to models/pattern_index/ for fast nearest-neighbor retrieval at inference.

Run: python train/build_pattern_index.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PATTERNS_FILE = "data/patterns/scam_patterns.jsonl"
INDEX_DIR = Path("models/pattern_index")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_patterns(path: str) -> list:
    patterns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                patterns.append(json.loads(line))
    return patterns


def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    patterns = load_patterns(PATTERNS_FILE)
    print(f"[OK] Loaded {len(patterns)} patterns")

    # Build embedding text: combine pattern name + example for richer representation
    texts = [f"{p['pattern']}. {p['example']}" for p in patterns]

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Inner-product on normalized vectors == cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(INDEX_DIR / "patterns.faiss"))
    with open(INDEX_DIR / "patterns.json", "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2, ensure_ascii=False)
    with open(INDEX_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({"embed_model": EMBED_MODEL, "dim": dim, "count": len(patterns)}, f, indent=2)

    print(f"[OK] Index built with {len(patterns)} patterns")
    print(f"[OK] Saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
