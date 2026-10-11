"""FastAPI REST interface for Athena RAG system with PDF upload support."""

import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from athena.graph import build_graph
from athena.ingestion import extract_pdf_pages, ingest_documents
from athena.tools import build_vectorstore

app = FastAPI(
    title="Athena RAG",
    description="Production-grade self-correcting Retrieval-Augmented Generation system with PDF ingestion",
    version="0.1.0",
)

graph = build_graph()


class QueryRequest(BaseModel):
    """Schema for query requests."""

    question: str = Field(..., min_length=3, max_length=1000)
    max_rewrites: int = Field(default=2, ge=0, le=5)


class IngestRequest(BaseModel):
    """Schema for direct document JSON ingestion."""

    documents: list[dict]


@app.get("/health")
async def health():
    """Readiness and liveness probe."""
    return {"status": "ok"}


@app.post("/query")
async def query(req: QueryRequest):
    """Run self-correcting RAG workflow for incoming query."""
    initial_state = {
        "question": req.question,
        "retrieval_query": req.question,
        "rewrite_count": 0,
        "max_rewrites": req.max_rewrites,
        "trace": [],
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
        "rewrites": result.get("rewrite_count", 0),
        "trace": result.get("trace", []),
    }


@app.post("/ingest")
async def ingest(req: IngestRequest):
    """Ingest raw document dictionaries into Chroma vector store."""
    try:
        build_vectorstore(req.documents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ingested", "count": len(req.documents)}


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    """
    Upload and index any PDF (supports 100+ pages) into the vector store.
    Extracts text page by page with source and page number metadata.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF document.")

    try:
        pdf_bytes = await file.read()
        docs = extract_pdf_pages(
            pdf_bytes,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not docs:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in PDF. (Ensure it contains text, not scanned images only)",
            )

        count = ingest_documents(docs)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_ingested": count,
            "message": f"Successfully indexed {count} passages from {file.filename}.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(exc)}")
