"""
download_models.py — One-time model and data download script.
Run: python download_models.py
"""

import subprocess
import sys
import os
from pathlib import Path

MODELS_DIR = Path("models")
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def check_ollama():
    """Verify Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            print("[!] Ollama is installed but the server may not be running.")
            print("    Start it with: ollama serve")
            return False
        print("[OK] Ollama is running.")
        return True
    except FileNotFoundError:
        print("[!] Ollama is not installed.")
        print("    Install from: https://ollama.com/download")
        return False


def pull_ollama_model(model_name: str):
    """Pull an Ollama model if not already present."""
    print(f"[...] Pulling Ollama model: {model_name}")
    result = subprocess.run(
        ["ollama", "pull", model_name],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[OK] {model_name} ready.")
    else:
        print(f"[!] Failed to pull {model_name}: {result.stderr}")


def download_hf_models():
    """Download sentence-transformer and classifier models."""
    from sentence_transformers import SentenceTransformer

    models = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "tactic-classifier": "distilbert-base-uncased",  # base for fine-tuning
    }

    for local_name, hf_name in models.items():
        print(f"[...] Downloading {hf_name}")
        model = SentenceTransformer(hf_name) if "MiniLM" in hf_name else None
        if model:
            save_path = MODELS_DIR / local_name
            model.save(str(save_path))
            print(f"[OK] Saved to {save_path}")


def download_datasets():
    """Download public scam/phishing datasets."""
    import urllib.request

    datasets = {
        # SMS Spam Collection (Kaggle-format: v1=label, v2=text)
        # justmarkham/python-ds URL is dead; this is a working mirror.
        "sms_spam.csv": (
            "https://raw.githubusercontent.com/mohitgupta-omg/"
            "Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        ),
        # Nazario corpus — HF mirror stores Parquet, not CSV;
        # converted to nazario.csv below.
        "nazario.parquet": (
            "https://huggingface.co/datasets/puyang2025/"
            "phish-email-datasets/resolve/main/Nazario.parquet"
        ),
    }

    for filename, url in datasets.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"[OK] {filename} already exists.")
            continue
        print(f"[...] Downloading {filename}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[OK] Saved to {dest}")
        except Exception as e:
            print(f"[!] Failed to download {filename}: {e}")
            print(f"    Download manually from: {url}")

    # Convert Nazario parquet -> CSV for easy hand-labeling
    pq = DATA_DIR / "nazario.parquet"
    csv = DATA_DIR / "nazario.csv"
    if pq.exists() and not csv.exists():
        try:
            import pandas as pd
            pd.read_parquet(pq).to_csv(csv, index=False)
            print(f"[OK] Converted {pq.name} -> {csv.name}")
        except Exception as e:
            print(f"[!] Parquet conversion failed: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Second Look — Model & Data Setup")
    print("=" * 50)

    if not check_ollama():
        print("\n[!] Install Ollama first, then re-run this script.")
        sys.exit(1)

    # Recommended model for constrained generation (fast, low VRAM)
    pull_ollama_model("qwen3:4b-instruct-2507-q8_0")

    download_hf_models()
    download_datasets()

    print("\n[DONE] Setup complete. Run 'streamlit run app.py' to start.")
