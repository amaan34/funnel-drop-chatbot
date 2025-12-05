from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Configuration for the embedding model used to build the vector store."""

    model_name: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 64
    max_input_tokens: int = 8191


# Default config targeting OpenAI text-embedding-3-small
DEFAULT_EMBEDDING_CONFIG = EmbeddingModelConfig()
