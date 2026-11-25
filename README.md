# Enterprise HR Policy RAG & LLMOps Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application built to answer employee policy questions across corporate HR handbooks with **hybrid retrieval (`BM25` + dense vector search), cross-encoder reranking (`FlashRank`), semantic caching, input guardrails, multi-layer fallback handling, and real-time `₹` cost/latency tracking**—all accessible through `app.py` with embedded pipeline execution and optional FastAPI integration.

---

## Visual Proof & System Walkthrough

### 1. 40-Query Benchmark Evaluation (CLI)
![40-Query Evaluation Summary](screenshots/eval_40_queries.png)

### 2. Automated Unit & API Test Suite (`pytest`)
![Pytest Results](screenshots/pytest_results.png)

### 3. FastAPI Backend (`Swagger /docs`)
![FastAPI Docs](screenshots/fastapi_docs.png)

### 4. Streamlit Application (`app.py` — Live Metrics, Benchmark & Citations)
![Streamlit Application](screenshots/streamlit_ui.png)

---

## Verified 40-Query Evaluation Results

Benchmarked via `python -m src.evals` across **35 in-domain HR policy questions** (5 handbooks) and **5 out-of-scope adversarial queries**:

| Metric | Baseline | Production Pipeline | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Retrieval Accuracy** | 80.0% | **82.9%** | **+2.9%** accuracy gain via hybrid `BM25` + vector search and `FlashRank` reranking |
| **Out-of-Scope Defense** | — | **80.0%** (4/5 blocked) | Guardrails + low-relevance/abstention fallback reduce off-topic hallucinations |
| **Median End-to-End Latency** | 10.01 s *(LLM rewrite)* | **4.13 s** | **-58.7%** latency reduction via deterministic HR synonym expansion & context cap |
| **Average Cost per Query** | ₹0.0272 | **₹0.0271** (`₹0.0000` cached) | Under 3 paise per uncached query; `₹0.0000` on semantic cache hits |

---

## Architecture & Engineering Highlights

- **Unified Application Entrypoint (`app.py`):** Runs the RAG backend and Streamlit UI seamlessly in a single deployment (`EnterpriseRAGPipeline` in-process when standalone, or connected to FastAPI when `/health` is reachable), keeping Streamlit Community Cloud deployment simple with zero external backend cold-start dependency.
- **Hybrid Retrieval & Cross-Encoder Reranking (`src/retriever.py`):** Combines dense embeddings (`BAAI/bge-small-en-v1.5` with ChromaDB) and lexical `BM25` retrieval (`top_k=8`), reranked with `FlashRank` (`ms-marco-MiniLM-L-12-v2`) and surfaced with `[Document, Page]` citations.
- **Zero-Token Synonym Query Expansion (`src/pipeline.py`):** Replaced LLM query rewriting with deterministic HR synonym expansion (`HR_SYNONYM_MAP`) and a 1,200-character context cap to avoid daily token bottlenecks and cut median response time to **4.13s**.
- **Semantic Caching (`src/cache.py`):** Reuses answers for semantically similar/paraphrased questions at **`₹0.0000`** LLM cost while updating live cache-hit metrics in the UI.
- **Guardrails & Resilience (`src/guardrails.py`, `src/pipeline.py`):** Filters out-of-scope prompts, handles `429` rate limits with backoff (`7s` -> `15s`), falls back from `openai/gpt-oss-20b` to `openai/gpt-oss-120b`, and abstains when context relevance is insufficient.
- **Observability, Testing & CI/CD:** Instrumented with `Langfuse` tracing, verified with `pytest` (`tests/test_pipeline.py`), containerized with `Dockerfile` and `docker-compose.yml`, and automated via GitHub Actions (`.github/workflows/ci.yml`).

---

## Repository Structure

```text
enterprise-rag-system/
├── app.py                   # Streamlit UI + integrated RAG pipeline execution
├── src/
│   ├── pipeline.py          # End-to-end RAG orchestration, caching, fallbacks & metrics
│   ├── retriever.py         # Hybrid BM25 + ChromaDB retrieval & FlashRank reranking
│   ├── guardrails.py        # Input validation & out-of-scope filtering
│   ├── cache.py             # Semantic response cache
│   ├── ingestion.py         # PDF loading, chunking & vector index creation
│   └── evals.py             # 40-query evaluation harness
├── tests/
│   └── test_pipeline.py     # Unit & API tests (pytest)
├── screenshots/             # Evaluation, pytest, API & UI verification images
├── Dockerfile               # Container image definition
├── docker-compose.yml       # Local container setup
├── pyproject.toml           # Pytest & project configuration
└── .github/workflows/ci.yml # Automated CI test workflow
## Quickstart

```bash
# 1. Activate virtual environment & install dependencies
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Create local .env from template and add GROQ_API_KEY
copy .env.example .env

# 3. Run the Streamlit application (frontend + pipeline)
streamlit run app.py

# 4. Run unit tests and the 40-query benchmark
python -m pytest
python -m src.evals