"""Streamlit frontend for the Enterprise HR Policy RAG System (API + Embedded Fallback)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

# Sync Streamlit Cloud secrets into os.environ BEFORE importing src modules
for _key in (
    "GROQ_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "USD_TO_INR_RATE",
):
    try:
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
    except Exception:
        # Ignore when running locally without a .streamlit/secrets.toml file
        pass

from src.ingestion import run_ingestion  # noqa: E402
from src.pipeline import EnterpriseRAGPipeline  # noqa: E402

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Enterprise HR Policy RAG Assistant",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner="Initializing HR RAG pipeline and vector index...")
def _get_embedded_pipeline() -> EnterpriseRAGPipeline:
    """Load EnterpriseRAGPipeline in-process when external FastAPI is unavailable."""
    return EnterpriseRAGPipeline()


def _run_embedded_ingest(pipeline: EnterpriseRAGPipeline) -> dict[str, Any]:
    """Run ingestion via pipeline.ingest() if available, otherwise call run_ingestion()."""
    if hasattr(pipeline, "ingest") and callable(pipeline.ingest):
        return pipeline.ingest()
    return run_ingestion()


def _check_api_health() -> bool:
    """Return True if the external FastAPI backend is reachable and healthy."""
    if not API_BASE_URL:
        return False
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{API_BASE_URL}/health")
            return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception:
        return False


def _fetch_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Helper to call FastAPI endpoints safely."""
    url = f"{API_BASE_URL}{path}"
    with httpx.Client(timeout=timeout) as client:
        if method.upper() == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _render_badges(meta: dict[str, Any]) -> None:
    """Render cache, fallback, model, latency, and INR cost badges under an assistant reply."""
    cache_badge = "⚡ Cache Hit" if meta.get("cache_hit") else "🔎 Cache Miss"
    fallback_badge = "  •  🛡️ **Fallback**" if meta.get("fallback_used") else ""
    model_used = meta.get("model_used", "unknown")
    latency_ms = meta.get("latency_ms", 0.0)
    cost_inr = float(meta.get("cost_inr", 0.0))
    st.caption(
        f"{cache_badge}{fallback_badge}  •  **Model:** `{model_used}`  •  "
        f"**Latency:** `{latency_ms} ms`  •  **Cost:** `₹{cost_inr:.4f}`"
    )


def _render_sources(sources: list[dict[str, Any]]) -> None:
    """Render expandable source chunks with filename, page number, and score."""
    if not sources:
        return
    with st.expander(f"📚 View Source Chunks ({len(sources)})"):
        for idx, src in enumerate(sources, start=1):
            source_name = src.get("source", "Unknown PDF")
            page_num = src.get("page", "?")
            score = src.get("relevance_score", 0.0)
            content = src.get("content", "")
            st.markdown(f"**{idx}. `{source_name}` — Page {page_num}** *(Score: {score})*")
            st.text(content)
            if idx < len(sources):
                st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []

api_online = _check_api_health()

st.title("📄 Enterprise HR Policy RAG Assistant")
st.write(
    "Ask policy questions across your HR handbooks with hybrid retrieval, reranking, "
    "semantic caching, fallbacks, and `₹` cost tracking."
)

with st.sidebar:
    st.header("System Status & Controls")
    if api_online:
        st.caption(f"Backend Mode: `FastAPI ({API_BASE_URL})`")
        st.markdown("**API Status:** 🟢 Healthy")
    else:
        st.caption("Backend Mode: `Embedded Pipeline (Standalone)`")
        st.markdown("**Runtime Status:** 🟢 Active (In-Process)")

    data_path = Path("data")
    if data_path.exists():
        pdf_files = sorted(p.name for p in data_path.glob("*.pdf"))
        st.markdown(f"**HR PDFs in `data/`:** {len(pdf_files)}")
        if pdf_files:
            with st.expander("View Indexed PDFs"):
                for fname in pdf_files:
                    st.write(f"• {fname}")

    st.divider()

    st.subheader("📊 Live Metrics")
    try:
        if api_online:
            metrics = _fetch_json("GET", "/metrics", timeout=5.0)
        elif "embedded_pipeline" in st.session_state:
            metrics = st.session_state["embedded_pipeline"].metrics.snapshot()
        else:
            metrics = {
                "total_requests": 0,
                "cache_hit_rate_pct": 0.0,
                "avg_latency_ms": 0.0,
                "total_cost_inr": 0.0,
            }

        col1, col2 = st.columns(2)
        col1.metric("Requests", metrics.get("total_requests", 0))
        col2.metric("Cache Hit %", f"{metrics.get('cache_hit_rate_pct', 0.0):.1f}%")
        col3, col4 = st.columns(2)
        col3.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0.0):.0f} ms")
        col4.metric("Total Cost", f"₹{float(metrics.get('total_cost_inr', 0.0)):.4f}")
    except Exception as exc:
        st.info(f"Metrics unavailable: {exc}")

    st.divider()

    st.subheader("🎯 Verified Eval Benchmark (40 Queries)")
    b1, b2 = st.columns(2)
    b1.metric("Reranked Acc.", "82.9%")
    b2.metric("Fallback Acc.", "80.0%")
    b3, b4 = st.columns(2)
    b3.metric("Median Latency", "4.13 s")
    b4.metric("Avg Cost/Query", "₹0.0271")
    st.caption("Hybrid + FlashRank + Synonym Expansion • Run CLI: `python -m src.evals`")

    st.divider()

    if st.button("🧹 Clear Semantic Cache", use_container_width=True):
        try:
            if api_online:
                res = _fetch_json("POST", "/cache/clear", timeout=15.0)
                st.success(res.get("message", "Cache cleared."))
            else:
                pipeline = _get_embedded_pipeline()
                st.session_state["embedded_pipeline"] = pipeline
                pipeline.cache.clear()
                st.success("Semantic cache cleared.")
        except Exception as exc:
            st.error(f"Failed to clear cache: {exc}")

    if st.button("🔄 Re-Ingest PDFs", use_container_width=True):
        with st.spinner("Re-indexing PDFs..."):
            try:
                if api_online:
                    summary = _fetch_json("POST", "/ingest", timeout=120.0)
                else:
                    pipeline = _get_embedded_pipeline()
                    st.session_state["embedded_pipeline"] = pipeline
                    summary = _run_embedded_ingest(pipeline)
                st.success(
                    f"Indexed {summary.get('files_indexed', 0)} PDFs ({summary.get('total_chunks', 0)} chunks)."
                )
            except Exception as exc:
                st.error(f"Re-ingestion failed: {exc}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_badges(msg.get("meta", {}))
            _render_sources(msg.get("sources", []))

user_query = st.chat_input("Ask about leave policies, overtime, remote work, code of conduct...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching handbooks and generating grounded answer..."):
            try:
                if api_online:
                    data = _fetch_json(
                        "POST",
                        "/query",
                        payload={"question": user_query, "session_id": "streamlit-session"},
                        timeout=60.0,
                    )
                else:
                    pipeline = _get_embedded_pipeline()
                    st.session_state["embedded_pipeline"] = pipeline
                    data = pipeline.run(
                        question=user_query,
                        session_id="streamlit-session",
                    )

                answer = data.get("answer", "No response returned.")
                sources = data.get("sources", [])
                meta = {
                    "cache_hit": data.get("cache_hit", False),
                    "fallback_used": data.get("fallback_used", False),
                    "model_used": data.get("model_used", "unknown"),
                    "latency_ms": data.get("latency_ms", 0.0),
                    "cost_inr": data.get("cost_inr", 0.0),
                }

                st.markdown(answer)
                _render_badges(meta)
                _render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "meta": meta,
                    }
                )
            except Exception as exc:
                err_msg = f"Query failed: {exc}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err_msg,
                        "sources": [],
                        "meta": {
                            "cache_hit": False,
                            "fallback_used": False,
                            "model_used": "error",
                            "latency_ms": 0.0,
                            "cost_inr": 0.0,
                        },
                    }
                )