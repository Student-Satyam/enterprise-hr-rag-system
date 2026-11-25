"""Hybrid retriever combining ChromaDB vector search, BM25 keyword search, and FlashRank reranking."""

import logging
import pickle
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import get_settings
from src.ingestion import ensure_index_exists, get_embedding_model

logger = logging.getLogger(__name__)

# Attempt import across LangChain versions (langchain_classic in newer releases, langchain.retrievers in older)
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover
    from langchain.retrievers import EnsembleRetriever  # type: ignore[no-redef]


class SourceChunk(BaseModel):
    """Structured schema for a retrieved chunk returned to the pipeline and API."""

    source: str
    page: int
    content: str
    relevance_score: float


class HybridRerankRetriever:
    """Runs hybrid retrieval (Vector + BM25) followed by cross-encoder reranking via FlashRank."""

    def __init__(self, top_k: int = 8, top_n: int | None = None) -> None:
        settings = get_settings()
        self.top_k = top_k if top_k is not None else 8
        self.top_n = top_n or settings.reranker_top_n

        # Make sure Chroma and BM25 artifacts exist before initializing retrievers
        ensure_index_exists()

        try:
            # 1. Dense semantic retriever from ChromaDB
            embeddings = get_embedding_model()
            vector_store = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=embeddings,
                persist_directory=settings.vector_store_dir,
            )
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": self.top_k})

            # 2. Sparse lexical retriever using BM25 over pickled chunks
            with open(settings.bm25_docs_path, "rb") as f:
                bm25_docs: list[Document] = pickle.load(f)

            bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            bm25_retriever.k = self.top_k

            # 3. Combine both retrievers equally (0.5 vector + 0.5 BM25) using Reciprocal Rank Fusion
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5],
            )

            # 4. Initialize FlashRank cross-encoder reranker to refine candidate ordering
            self.reranker = FlashrankRerank(
                model=settings.reranker_model_name,
                top_n=self.top_n,
            )
        except Exception as exc:
            logger.error("Failed to initialize HybridRerankRetriever: %s", exc)
            raise

    def retrieve_multi(self, queries: list[str], top_k: int | None = None) -> list[SourceChunk]:
        """Retrieve candidate chunks across multiple query rewrites, deduplicate, and rerank."""
        limit = top_k or self.top_n
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return []

        try:
            seen_keys: set[tuple[str, int, str]] = set()
            merged_candidates: list[Document] = []

            for q in clean_queries:
                candidate_docs: list[Document] = self.ensemble_retriever.invoke(q)
                for doc in candidate_docs:
                    dedup_key = (
                        str(doc.metadata.get("source", "unknown.pdf")),
                        int(doc.metadata.get("page", 1)),
                        doc.page_content.strip(),
                    )
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        merged_candidates.append(doc)

            if not merged_candidates:
                return []

            # Rerank all deduplicated candidates against the primary user query
            reranked_docs: list[Document] = self.reranker.compress_documents(
                documents=merged_candidates,
                query=clean_queries[0],
            )

            results: list[SourceChunk] = []
            for doc in reranked_docs[:limit]:
                raw_score = doc.metadata.get("relevance_score", 0.0)
                score = round(float(raw_score), 4) if raw_score is not None else 0.0

                results.append(
                    SourceChunk(
                        source=str(doc.metadata.get("source", "unknown.pdf")),
                        page=int(doc.metadata.get("page", 1)),
                        content=doc.page_content,
                        relevance_score=score,
                    )
                )

            return results
        except Exception as exc:
            logger.error("Multi-query retrieval failed for '%s': %s", clean_queries[0], exc)
            raise

    def retrieve(self, query: str, top_k: int | None = None) -> list[SourceChunk]:
        """Retrieve candidate chunks via hybrid search and rerank to top_n results."""
        return self.retrieve_multi([query], top_k=top_k)