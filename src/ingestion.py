"""PDF ingestion pipeline for building ChromaDB vectors and BM25 lexical index."""

import logging
import pickle
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings

logger = logging.getLogger(__name__)


def get_embedding_model() -> FastEmbedEmbeddings:
    """Initialize lightweight ONNX FastEmbed model (avoids heavy PyTorch RAM overhead)."""
    settings = get_settings()
    return FastEmbedEmbeddings(model_name=settings.embedding_model_name)


def load_pdf_documents(data_dir: str) -> list[Document]:
    """Load all PDF files in data_dir and normalize metadata (filename + 1-based page)."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path.resolve()}")

    # Sort filenames so ingestion order is deterministic across environments
    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{data_path.resolve()}'.")

    all_pages: list[Document] = []

    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            for page_doc in pages:
                # PyPDFLoader uses 0-based 'page' metadata; store 1-based page for human citations
                raw_page = page_doc.metadata.get("page", 0)
                human_page = int(raw_page) + 1 if isinstance(raw_page, int) else 1

                # Keep only the clean filename (not full local/container path)
                page_doc.metadata = {
                    "source": pdf_file.name,
                    "page": human_page,
                }
                all_pages.append(page_doc)

            logger.info("Loaded %d pages from %s", len(pages), pdf_file.name)
        except Exception as exc:
            logger.error("Failed to load PDF '%s': %s", pdf_file.name, exc)
            raise

    return all_pages


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split page documents into overlapping chunks while keeping metadata attached."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer splitting at paragraph/sentence boundaries before falling back to characters
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Filter out empty/whitespace-only chunks that add noise to retrieval
    non_empty_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    logger.info("Created %d non-empty chunks from %d pages", len(non_empty_chunks), len(documents))
    return non_empty_chunks


def run_ingestion(
    data_dir: str | None = None,
    persist_dir: str | None = None,
) -> dict[str, Any]:
    """Run full ingestion: load PDFs -> chunk -> index in ChromaDB -> save BM25 docs."""
    settings = get_settings()
    target_data_dir = data_dir or settings.data_dir
    target_persist_dir = Path(persist_dir or settings.vector_store_dir)
    target_persist_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pdf_documents(target_data_dir)
    chunks = split_documents(
        documents=pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embeddings = get_embedding_model()

    try:
        # Persist dense embeddings in local ChromaDB
        vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=str(target_persist_dir),
        )
        # Clear existing records before re-indexing to avoid duplicate chunks on re-ingest
        existing = vector_store.get()
        if existing and existing.get("ids"):
            vector_store.delete(ids=existing["ids"])

        vector_store.add_documents(chunks)

        # Save serialized chunks so BM25Retriever can build its keyword index fast on startup
        bm25_path = target_persist_dir / settings.bm25_docs_filename
        with open(bm25_path, "wb") as f:
            pickle.dump(chunks, f)

    except Exception as exc:
        logger.error("Error persisting vector store or BM25 index: %s", exc)
        raise

    unique_sources = sorted({c.metadata.get("source", "unknown.pdf") for c in chunks})
    summary = {
        "status": "success",
        "files_indexed": len(unique_sources),
        "sources": unique_sources,
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "persist_dir": str(target_persist_dir),
    }
    logger.info("Ingestion complete: %s", summary)
    return summary


def ensure_index_exists() -> None:
    """Build the index automatically if missing (essential for Render ephemeral disk)."""
    settings = get_settings()
    chroma_sqlite = settings.vector_store_path / "chroma.sqlite3"
    bm25_file = settings.bm25_docs_path

    if chroma_sqlite.exists() and bm25_file.exists():
        logger.info("Existing vector store and BM25 index found at '%s'.", settings.vector_store_dir)
        return

    logger.warning("Index not found at '%s'. Running automatic ingestion...", settings.vector_store_dir)
    run_ingestion(
        data_dir=settings.data_dir,
        persist_dir=settings.vector_store_dir,
    )