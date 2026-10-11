"""Document ingestion pipeline for Athena RAG with PDF and multi-page support."""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, BinaryIO

from pypdf import PdfReader

from athena.observability import log_event, setup_logger
from athena.tools.vectorstore import build_vectorstore

logger = setup_logger("athena.ingestion")


def load_documents_from_json(file_path: str | Path) -> list[dict[str, Any]]:
    """Load documents from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of documents in {path}, got {type(data).__name__}")

    return data


def extract_pdf_pages(
    file_source: str | Path | bytes | BinaryIO,
    filename: str = "document.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """
    Extract text and chunk pages from a PDF file (supports 100+ pages).
    Preserves page numbers in metadata and citations.
    """
    if isinstance(file_source, (str, Path)):
        reader = PdfReader(str(file_source))
        source_name = Path(file_source).name
    elif isinstance(file_source, bytes):
        reader = PdfReader(io.BytesIO(file_source))
        source_name = filename
    else:
        reader = PdfReader(file_source)
        source_name = filename

    total_pages = len(reader.pages)
    documents = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Failed to extract text from page {idx} of {source_name}: {e}")
            page_text = ""

        page_text = page_text.strip()
        if not page_text:
            continue

        # If page text is longer than chunk size, chunk it with overlap
        chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_idx, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "source": f"{source_name} (Page {idx})",
                "metadata": {
                    "filename": source_name,
                    "page": idx,
                    "total_pages": total_pages,
                    "chunk_index": chunk_idx,
                },
            })

    log_event(
        logger,
        "pdf_extracted",
        filename=source_name,
        total_pages=total_pages,
        total_chunks=len(documents),
    )
    return documents


def load_document_from_file(
    file_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    """Load documents from a PDF, text, markdown, or JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_pages(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if suffix == ".json":
        return load_documents_from_json(path)

    # For .txt, .md, and other text documents
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"File {path.name} is empty.")

    chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        {
            "source": path.name,
            "content": c,
            "metadata": {"file_path": str(path), "chunk_index": i},
        }
        for i, c in enumerate(chunks)
    ]


def load_documents_from_directory(
    dir_path: str | Path,
    extensions: tuple = (".pdf", ".txt", ".md", ".json"),
) -> list[dict[str, Any]]:
    """Load all matching documents from a directory."""
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {path}")

    all_docs = []
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            try:
                docs = load_document_from_file(file)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Could not load {file.name}: {e}")
    return all_docs


def validate_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Validate that a document has required fields: content and source."""
    if not isinstance(doc, dict):
        raise ValueError(f"Document must be a dictionary, got {type(doc).__name__}")
    if "content" not in doc or not doc["content"]:
        raise ValueError("Document missing non-empty 'content' field")

    return {
        "content": str(doc["content"]).strip(),
        "source": str(doc.get("source", "unknown")).strip(),
        "metadata": doc.get("metadata", {}),
    }


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Character/token window chunker with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def ingest_documents(
    docs: list[dict[str, Any]],
    persist_dir: str = "./chroma_db",
    auto_chunk: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """Ingest a list of documents into the vector store."""
    validated_docs = [validate_document(d) for d in docs]

    if auto_chunk:
        chunked_docs = []
        for doc in validated_docs:
            chunks = chunk_text(doc["content"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    "content": chunk,
                    "source": doc["source"],
                    "metadata": {**doc.get("metadata", {}), "chunk_index": idx},
                })
        final_docs = chunked_docs
    else:
        final_docs = validated_docs

    build_vectorstore(final_docs, persist_dir=persist_dir)
    log_event(logger, "ingest_completed", count=len(final_docs), persist_dir=persist_dir)
    return len(final_docs)


def main():
    """CLI runner for document and PDF ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents or PDFs into Athena RAG vector store.")
    parser.add_argument("path", help="Path to a file (.pdf, .txt, .md, .json) or directory to ingest")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters")

    args = parser.parse_args()
    target_path = Path(args.path)

    if not target_path.exists():
        print(f"Error: Path '{target_path}' does not exist.")
        sys.exit(1)

    if target_path.is_dir():
        docs = load_documents_from_directory(target_path)
    else:
        docs = load_document_from_file(
            target_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

    print(f"Loaded {len(docs)} document chunks. Ingesting into ChromaDB vector store...")
    count = ingest_documents(docs)
    print(f"Successfully ingested {count} chunk(s) from '{target_path.name}' into ChromaDB!")


if __name__ == "__main__":
    main()
