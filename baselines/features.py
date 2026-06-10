"""Feature extraction utilities."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def build_tfidf_pipeline(estimator, tfidf_config: dict) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        min_df=tfidf_config["min_df"],
    )
    return Pipeline([("tfidf", vectorizer), ("model", estimator)])


def _safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def get_text_embeddings(
    texts: list[str],
    subject_ids: list[str],
    embedding_config: dict,
    project_root: Path,
) -> np.ndarray:
    """Encode transcripts with the configured embedding model and cache the matrix."""
    provider = embedding_config.get("provider", "sentence_transformers")
    model_name = embedding_config["model_name"]
    cache_dir = project_root / embedding_config.get("cache_dir", "results/embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_model_name(f"{provider}_{model_name}")
    embeddings_path = cache_dir / f"{safe_name}_train.npy"
    metadata_path = cache_dir / f"{safe_name}_train.json"

    if embeddings_path.exists() and metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        if (
            metadata.get("provider") == provider
            and metadata.get("model_name") == model_name
            and metadata.get("subject_ids") == subject_ids
            and metadata.get("normalize_embeddings")
            == embedding_config.get("normalize_embeddings", True)
        ):
            return np.load(embeddings_path)

    if provider == "sentence_transformers":
        embeddings = _encode_sentence_transformers(texts, embedding_config)
    elif provider == "openai":
        embeddings = _encode_openai(texts, embedding_config)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embedding_config.get("normalize_embeddings", True):
        embeddings = _l2_normalize(embeddings)

    np.save(embeddings_path, embeddings)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "provider": provider,
                "model_name": model_name,
                "subject_ids": subject_ids,
                "shape": list(embeddings.shape),
                "normalize_embeddings": embedding_config.get(
                    "normalize_embeddings", True
                ),
            },
            f,
            indent=2,
        )

    return embeddings


def _encode_sentence_transformers(
    texts: list[str], embedding_config: dict
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_config["model_name"])
    return model.encode(
        texts,
        batch_size=embedding_config.get("batch_size", 8),
        normalize_embeddings=False,
        show_progress_bar=True,
    )


def _encode_openai(texts: list[str], embedding_config: dict) -> np.ndarray:
    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or run "
            "`export OPENAI_API_KEY=...` first."
        )

    client = OpenAI()
    model_name = embedding_config["model_name"]
    batch_size = embedding_config.get("batch_size", 64)
    vectors = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model_name, input=batch)
        vectors.extend(item.embedding for item in response.data)

    return np.asarray(vectors, dtype=np.float32)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms
