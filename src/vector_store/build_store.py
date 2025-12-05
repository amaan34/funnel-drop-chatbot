import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from src.embeddings import EmbeddingGenerator, EmbeddingModelConfig, EmbeddingOptimizer
from src.vector_store.bm25_store import BM25KeywordStore
from src.vector_store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def load_chunks(input_path: str = "data/processed_chunks.json") -> List[Dict]:
    """Load processed chunks from disk."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed chunks not found at {input_path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def persist_embedded_chunks(chunks: List[Dict], output_path: str = "data/embedded_chunks.json") -> None:
    """Write embedded chunks to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info("Embedded chunks written to %s", output_path)


def build_stores(
    *,
    chunks_path: str = "data/processed_chunks.json",
    chroma_directory: str = "data/vector_store",
    chroma_collection: str = "funnel_drop_chunks",
    bm25_index_path: str = "data/bm25_index.pkl",
    embedding_model: str = "text-embedding-3-small",
    force_recreate: bool = False,
) -> Tuple[ChromaVectorStore, BM25KeywordStore, List[Dict]]:
    """
    Build the Chroma vector store and BM25 index from processed chunks.
    """
    load_dotenv()

    chunks = load_chunks(chunks_path)
    logger.info("Loaded %s chunks from %s", len(chunks), chunks_path)

    config = EmbeddingModelConfig(model_name=embedding_model)
    embedder = EmbeddingGenerator(config=config)
    optimized_texts = EmbeddingOptimizer.prepare_chunks_for_embedding(chunks)
    embeddings = embedder.generate_embeddings_batch(optimized_texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding generation failed; counts do not match.")

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        metadata = chunk.get("metadata", {}) or {}
        metadata["embedding_model"] = embedding_model
        metadata["embedding_dimension"] = config.dimension
        chunk["metadata"] = metadata

    persist_embedded_chunks(chunks)

    chroma_store = ChromaVectorStore(persist_directory=chroma_directory)
    chroma_store.create_collection(
        collection_name=chroma_collection,
        embedding_dimension=config.dimension,
        force_recreate=force_recreate,
    )
    chroma_store.add_chunks(chunks)
    chroma_store.persist()

    bm25_store = BM25KeywordStore()
    bm25_store.build_index(chunks)
    bm25_store.save_index(bm25_index_path)

    logger.info("Vector store count: %s", chroma_store.count())
    logger.info("BM25 index saved to %s", bm25_index_path)
    return chroma_store, bm25_store, chunks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_stores()


if __name__ == "__main__":  # pragma: no cover - manual helper
    main()
