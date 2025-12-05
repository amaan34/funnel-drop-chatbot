import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Wrapper around ChromaDB for semantic retrieval."""

    def __init__(self, persist_directory: str = "data/vector_store") -> None:
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None
        logger.info("Initialized ChromaDB at %s", persist_directory)

    def create_collection(
        self,
        collection_name: str = "funnel_drop_chunks",
        embedding_dimension: int = 1536,
        force_recreate: bool = False,
    ):
        """Create or fetch a collection."""
        if force_recreate:
            try:
                self.client.delete_collection(collection_name)
                logger.info("Deleted existing collection %s", collection_name)
            except Exception:
                logger.debug("No existing collection %s to delete", collection_name)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Credit card onboarding funnel drop chunks",
                "embedding_dimension": embedding_dimension,
            },
        )
        logger.info("Collection '%s' ready (dim=%s)", collection_name, embedding_dimension)
        return self.collection

    def _flatten_metadata(self, metadata: Dict) -> Dict:
        """Flatten metadata for Chroma (accepts only primitives)."""
        flattened: Dict[str, str | int | float | bool] = {}
        for key, value in (metadata or {}).items():
            if isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif isinstance(value, list):
                flattened[key] = ",".join(map(str, value))
            elif value is None:
                flattened[key] = ""
            else:
                flattened[key] = str(value)
        return flattened

    def add_chunks(self, chunks: List[Dict]) -> None:
        """Add embedded chunks to the collection."""
        if not self.collection:
            raise ValueError("Collection not initialized. Call create_collection() first.")

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for chunk in chunks:
            if "embedding" not in chunk:
                raise ValueError(f"Chunk {chunk.get('chunk_id')} missing embedding.")
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["content"])
            metadatas.append(self._flatten_metadata(chunk.get("metadata", {})))

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Added %s chunks to Chroma collection %s", len(chunks), self.collection.name)
        self.persist()

    def persist(self) -> None:
        """Persist collection to disk when using a persistent client."""
        try:
            self.client.persist()
        except Exception:
            # Older Chroma versions may not require or support explicit persist.
            logger.debug("Chroma persist skipped or not supported.")

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> Dict:
        """Query by embedding."""
        if not self.collection:
            raise ValueError("Collection not initialized")

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

    def query_by_text(
        self,
        query_text: str,
        embedding_generator,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Query by raw text."""
        query_embedding = embedding_generator.generate_embedding(query_text)
        results = self.query(query_embedding=query_embedding, n_results=n_results, where=where)

        formatted: List[Dict] = []
        for i in range(len(results["ids"][0])):
            formatted.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )
        return formatted

    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a chunk by ID."""
        if not self.collection:
            raise ValueError("Collection not initialized")
        result = self.collection.get(ids=[chunk_id])
        if result["ids"]:
            return {
                "chunk_id": result["ids"][0],
                "content": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        return None

    def count(self) -> int:
        """Return the number of stored chunks."""
        if not self.collection:
            return 0
        return self.collection.count()

    def get_collection_stats(self) -> Dict:
        """Return simple stats for the collection."""
        if not self.collection:
            return {}
        count = self.count()
        sample = self.collection.get(limit=min(100, count))
        sections: Dict[str, int] = {}
        stages: Dict[str, int] = {}
        for metadata in sample["metadatas"]:
            section = metadata.get("section", "unknown")
            stage = metadata.get("stage", "general")
            sections[section] = sections.get(section, 0) + 1
            stages[stage] = stages.get(stage, 0) + 1
        return {
            "total_chunks": count,
            "sections": sections,
            "stages": stages,
            "collection_name": self.collection.name,
        }

