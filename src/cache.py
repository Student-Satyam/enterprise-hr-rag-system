"""Semantic cache backed by a dedicated ChromaDB collection using cosine similarity."""

import json
import logging
import uuid
from typing import Any

from langchain_chroma import Chroma

from src.config import get_settings
from src.ingestion import get_embedding_model

logger = logging.getLogger(__name__)


class SemanticCache:
    """Caches question-answer pairs and retrieves matches above a cosine similarity threshold."""

    def __init__(self, threshold: float | None = None) -> None:
        settings = get_settings()
        self.threshold = threshold if threshold is not None else settings.cache_similarity_threshold
        self.persist_dir = settings.vector_store_dir
        self.collection_name = "semantic_cache"
        self.embeddings = get_embedding_model()
        self._store = self._init_store()

    def _init_store(self) -> Chroma:
        """Initialize Chroma collection configured for cosine distance."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def get(self, query: str) -> dict[str, Any] | None:
        """Return cached payload if an existing query has cosine similarity >= self.threshold."""
        try:
            # similarity_search_with_score returns cosine distance in [0, 2] when hnsw:space=cosine
            results = self._store.similarity_search_with_score(query, k=1)
            if not results:
                return None

            doc, cosine_distance = results[0]
            similarity = 1.0 - float(cosine_distance)

            if similarity >= self.threshold:
                logger.info(
                    "Semantic cache HIT (similarity=%.4f >= %.2f)", similarity, self.threshold
                )
                return {
                    "answer": doc.metadata.get("answer", ""),
                    "sources": json.loads(doc.metadata.get("sources_json", "[]")),
                    "similarity": round(similarity, 4),
                }

            logger.info("Semantic cache MISS (best similarity=%.4f)", similarity)
            return None
        except Exception as exc:
            # Cache failures should never break user requests; fall back gracefully to live RAG
            logger.warning("Semantic cache lookup skipped due to error: %s", exc)
            return None

    def set(self, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        """Store a new query, its grounded answer, and serialized source chunks in Chroma."""
        try:
            self._store.add_texts(
                texts=[query],
                metadatas=[
                    {
                        "answer": answer,
                        "sources_json": json.dumps(sources),
                    }
                ],
                ids=[str(uuid.uuid4())],
            )
            logger.info("Stored response in semantic cache.")
        except Exception as exc:
            logger.warning("Failed to write to semantic cache: %s", exc)

    def clear(self) -> None:
        """Delete all entries in the semantic cache collection and recreate it cleanly."""
        try:
            existing = self._store.get()
            if existing and existing.get("ids"):
                self._store.delete(ids=existing["ids"])
            logger.info("Semantic cache cleared.")
        except Exception as exc:
            logger.error("Failed to clear semantic cache: %s", exc)
            raise