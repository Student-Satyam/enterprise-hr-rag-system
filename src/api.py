"""FastAPI service exposing health, metrics, query, ingestion, cache, and eval endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.evals import run_40_query_eval
from src.ingestion import ensure_index_exists, run_ingestion
from src.monitoring import MetricsTracker
from src.pipeline import EnterpriseRAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)


# --- Pydantic Schemas ---

class SourceChunk(BaseModel):
    source: str
    page: int
    content: str
    relevance_score: float = 0.0


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    model_used: str
    cache_hit: bool
    fallback_used: bool = False
    latency_ms: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    cost_inr: float = 0.0
    trace_id: str | None = None


# --- Application & Lifespan ---

metrics_tracker = MetricsTracker()
rag_pipeline: EnterpriseRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure vector/BM25 indexes exist on startup and initialize the RAG pipeline."""
    global rag_pipeline
    logger.info("Starting Enterprise RAG API...")
    try:
        ensure_index_exists()
        rag_pipeline = EnterpriseRAGPipeline(metrics_tracker=metrics_tracker)
        logger.info("Enterprise RAG pipeline ready.")
    except Exception as exc:
        logger.exception("Startup initialization failed: %s", exc)
        rag_pipeline = None
    yield
    logger.info("Shutting down Enterprise RAG API...")


app = FastAPI(
    title="Enterprise HR Policy RAG API",
    version="1.0.0",
    description="Hybrid Search + FlashRank + Semantic Cache + Guardrails + Model Routing + Fallbacks + INR Cost",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Liveness and index readiness check for Docker and Render."""
    settings = get_settings()
    index_ready = (
        settings.vector_store_path / "chroma.sqlite3"
    ).exists() and settings.bm25_docs_path.exists()

    return {
        "status": "ok" if (rag_pipeline is not None and index_ready) else "degraded",
        "pipeline_initialized": rag_pipeline is not None,
        "index_ready": index_ready,
    }


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Return live application monitoring metrics including INR cost and fallbacks."""
    return metrics_tracker.snapshot()


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest) -> QueryResponse:
    """Run an HR policy question through the full RAG pipeline."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized yet.")

    result = rag_pipeline.run(question=payload.question, session_id=payload.session_id)
    return QueryResponse(**result)


@app.post("/ingest")
def trigger_ingestion() -> dict[str, Any]:
    """Re-index PDFs from the data directory and refresh the retriever."""
    global rag_pipeline
    settings = get_settings()
    try:
        summary = run_ingestion(
            data_dir=settings.data_dir,
            persist_dir=settings.vector_store_dir,
        )
        rag_pipeline = EnterpriseRAGPipeline(metrics_tracker=metrics_tracker)
        return summary
    except Exception as exc:
        logger.exception("Ingestion endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/cache/clear")
def clear_semantic_cache() -> dict[str, str]:
    """Clear all entries from the semantic cache."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized yet.")
    try:
        rag_pipeline.cache.clear()
        return {"status": "cleared", "message": "Semantic cache cleared successfully."}
    except Exception as exc:
        logger.exception("Failed to clear semantic cache: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/eval")
def evaluate_rag() -> dict[str, Any]:
    """Run the 40-query RAG evaluation benchmark."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized yet.")
    try:
        return run_40_query_eval(pipeline=rag_pipeline)
    except Exception as exc:
        logger.exception("Evaluation endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc