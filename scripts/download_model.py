#!/usr/bin/env python3
"""Download the sentence-transformer model to the local models/ cache.

Run once after cloning the repo so that the app starts instantly
without fetching weights from HuggingFace on first request.

Usage:
    python scripts/download_model.py
"""

from __future__ import annotations

from pathlib import Path

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    from sentence_transformers import SentenceTransformer

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_NAME} to {MODELS_DIR} ...")
    SentenceTransformer(MODEL_NAME, cache_folder=str(MODELS_DIR))
    print("Done. Model cached locally.")


if __name__ == "__main__":
    main()
