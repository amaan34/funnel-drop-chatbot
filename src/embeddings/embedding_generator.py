import logging
import os
from typing import Any, Dict, Iterable, List

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from .embedding_config import DEFAULT_EMBEDDING_CONFIG, EmbeddingModelConfig

logger = logging.getLogger(__name__)


class EmbeddingOptimizer:
    """Utility to build richer text inputs prior to embedding."""

    @staticmethod
    def prepare_text_for_embedding(chunk: Dict[str, Any]) -> str:
        """
        Create an optimized text representation that emphasizes query-relevant signals.
        """
        metadata = chunk.get("metadata", {}) or {}
        chunk_type = metadata.get("chunk_type", "unknown")
        content = chunk.get("content", "")

        if chunk_type == "faq":
            question = metadata.get("question", "")
            optimized = f"{question} {question} {content}"
        elif chunk_type == "funnel_step":
            stage = metadata.get("stage_name") or metadata.get("stage") or ""
            optimized = f"{stage} {stage} onboarding step. {content}"
        elif chunk_type == "offers_summary":
            optimized = f"Credit card offers benefits rewards. {content}"
        elif chunk_type == "call_template":
            stage = metadata.get("stage") or ""
            optimized = f"Outreach template for {stage} stage dropoff. {content}"
        else:
            optimized = content

        if metadata.get("stage"):
            optimized = f"Related to {metadata['stage']} stage. {optimized}"

        max_length = 500
        if len(optimized) > max_length:
            optimized = optimized[:max_length]
        return optimized

    @staticmethod
    def prepare_chunks_for_embedding(chunks: List[Dict[str, Any]]) -> List[str]:
        """Prepare a list of chunk texts for embedding."""
        return [EmbeddingOptimizer.prepare_text_for_embedding(chunk) for chunk in chunks]


class EmbeddingGenerator:
    """Generates embeddings for text chunks using OpenAI embeddings."""

    def __init__(
        self,
        api_key: str | None = None,
        config: EmbeddingModelConfig = DEFAULT_EMBEDDING_CONFIG,
    ) -> None:
        self.config = config
        key = api_key or os.getenv("OPENAI_API_KEY")
        # Fail fast if the key is missing or still the placeholder from .env
        if not key or "YOUR_KEY" in key:
            raise ValueError(
                "OPENAI_API_KEY is required to generate embeddings. Update your .env with a valid key."
            )
        self.client = OpenAI(api_key=key)
        logger.info(
            "Initialized OpenAI embedding model %s (dim=%s)",
            self.config.model_name,
            self.config.dimension,
        )

    def _embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed a single batch of texts."""
        response = self.client.embeddings.create(
            model=self.config.model_name,
            input=list(texts),
        )
        return [record.embedding for record in response.data]

    def generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding."""
        return self._embed_batch([text])[0]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with batching and progress."""
        if not texts:
            return []

        embeddings: List[List[float]] = []
        batch_size = max(self.config.batch_size, 1)

        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[start : start + batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    def add_embeddings_to_chunks(
        self, chunks: List[Dict[str, Any]], optimize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Add an 'embedding' field to each chunk.

        Args:
            chunks: List of chunk dicts generated from step 2.
            optimize: Whether to apply text optimization before embedding.
        """
        if not chunks:
            return []

        texts = (
            EmbeddingOptimizer.prepare_chunks_for_embedding(chunks)
            if optimize
            else [c.get("content", "") for c in chunks]
        )
        embeddings = self.generate_embeddings_batch(texts)

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch. Got {len(embeddings)} embeddings for {len(chunks)} chunks."
            )

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
            metadata = chunk.get("metadata", {}) or {}
            metadata["embedding_model"] = self.config.model_name
            metadata["embedding_dimension"] = self.config.dimension
            chunk["metadata"] = metadata
        logger.info("Added embeddings to %s chunks", len(chunks))
        return chunks

    def save_embedded_chunks(self, chunks: List[Dict[str, Any]], output_path: str) -> None:
        """Save embedded chunks to disk as JSON."""
        import json
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info("Saved embedded chunks to %s", output_path)

