"""Tests for the Enterprise HR Policy RAG pipeline and FastAPI health endpoint."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from src.pipeline import EnterpriseRAGPipeline, _llm_indicates_no_answer
from src.retriever import SourceChunk


class _DummyCache:
    """In-memory stand-in for SemanticCache during unit tests."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    def get(self, query: str) -> dict[str, Any] | None:
        return self.store.get(query)

    def set(self, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        self.store[query] = {"answer": answer, "sources": sources}

    def clear(self) -> None:
        self.store.clear()


class _DummyRetriever:
    """In-memory stand-in for HybridRerankRetriever during unit tests."""

    def __init__(self, chunks: list[SourceChunk] | None = None) -> None:
        self.top_k = 8
        self._chunks = (
            chunks
            if chunks is not None
            else [
                SourceChunk(
                    source="Timelabs_HR_Policy.pdf",
                    page=4,
                    content="Regular employees scheduled to work 20+ hours per week for at least 1 year are eligible for unpaid leave.",
                    relevance_score=0.88,
                )
            ]
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[SourceChunk]:
        return self._chunks

    def retrieve_multi(self, queries: list[str], top_k: int | None = None) -> list[SourceChunk]:
        return self._chunks


def _make_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[SourceChunk] | None = None,
) -> EnterpriseRAGPipeline:
    monkeypatch.setattr(
        "src.pipeline.HybridRerankRetriever",
        lambda *args, **kwargs: _DummyRetriever(chunks),
    )
    monkeypatch.setattr(
        "src.pipeline.SemanticCache",
        lambda *args, **kwargs: _DummyCache(),
    )
    return EnterpriseRAGPipeline()


def test_synonym_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _make_pipeline(monkeypatch)
    rewrites = pipeline._rewrite_query_to_hr_formal(
        "What are the eligibility requirements for an unpaid leave of absence?",
        max_rewrites=1,
    )
    assert len(rewrites) == 1
    assert isinstance(rewrites[0], str) and len(rewrites[0].strip()) > 0


def test_answer_and_semantic_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _make_pipeline(monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "_generate_with_fallback",
        lambda primary_model, context_text, question, invoke_config: (
            "Regular employees working 20+ hours/week for at least 1 year qualify [Timelabs_HR_Policy.pdf, Page 4].",
            "openai/gpt-oss-20b",
            False,
            120,
            45,
        ),
    )

    first = pipeline.run(
        "What are the eligibility requirements for an unpaid leave of absence?",
        session_id="test-session",
    )
    assert first["cache_hit"] is False
    assert first["fallback_used"] is False
    assert "Timelabs_HR_Policy.pdf" in first["answer"]
    assert len(first["sources"]) == 1

    second = pipeline.run(
        "What are the eligibility requirements for an unpaid leave of absence?",
        session_id="test-session",
    )
    assert second["cache_hit"] is True
    assert second["cost_inr"] == 0.0
    assert second["model_used"] == "semantic-cache"


def test_low_relevance_and_llm_abstention_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    low_chunks = [
        SourceChunk(
            source="Timelabs_HR_Policy.pdf",
            page=1,
            content="General handbook overview.",
            relevance_score=0.05,
        )
    ]
    pipeline = _make_pipeline(monkeypatch, chunks=low_chunks)
    monkeypatch.setattr(
        pipeline,
        "_generate_with_fallback",
        lambda primary_model, context_text, question, invoke_config: (
            "I don't know — the provided HR handbooks do not cover that detail.",
            "openai/gpt-oss-20b",
            False,
            100,
            25,
        ),
    )

    assert _llm_indicates_no_answer("I don't know — the provided HR handbooks do not cover that detail.") is True
    res = pipeline.run("What is the company's asteroid mining bonus policy?", session_id="test-session")
    assert res["fallback_used"] is True
    assert res["model_used"] == "idk-fallback"
    assert "i don't know" in res["answer"].lower()


def test_guardrail_blocks_non_hr_query(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _make_pipeline(monkeypatch)
    res = pipeline.run(
        "What is the corporate discount code for Tesla Cybertruck purchases?",
        session_id="test-session",
    )
    assert res["model_used"] == "guardrail-blocked"
    assert res["sources"] == []


def test_fastapi_health_if_present() -> None:
    api_mod = None
    for mod_name in ("src.api", "src.main"):
        try:
            candidate = importlib.import_module(mod_name)
            if hasattr(candidate, "app"):
                api_mod = candidate
                break
        except Exception:
            continue

    if api_mod is None:
        pytest.skip("No FastAPI app found in src.api or src.main")

    from fastapi.testclient import TestClient

    client = TestClient(api_mod.app)
    resp = client.get("/health")
    assert resp.status_code == 200