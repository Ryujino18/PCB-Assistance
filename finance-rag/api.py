from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import chromadb
import traceback
from langchain_community.vectorstores import Chroma

from ingest import ingest_pdfs, DATA_DIR, CHROMA_DIR
from rag import get_answer

app = FastAPI(title="Finance RAG API")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Module-level state: track chunk count in memory to avoid holding ChromaDB file locks.
# Windows does NOT allow multiple processes/connections to hold the same file open.
_chunk_count: int = 0

class AskRequest(BaseModel):
    question: str
    top_k: int = 4

@app.on_event("startup")
async def startup_event():
    """Read existing chunk count once at startup, then release the connection immediately."""
    global _chunk_count
    if os.path.exists(CHROMA_DIR):
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            collection = client.get_or_create_collection("langchain")
            _chunk_count = collection.count()
            del client  # Release file handles immediately
        except Exception:
            _chunk_count = 0

@app.post("/index")
async def index_documents(files: List[UploadFile] = File(...)):
    global _chunk_count
    try:
        # Save uploaded files to the data directory using async read
        for file in files:
            file_location = os.path.join(DATA_DIR, file.filename)
            contents = await file.read()
            with open(file_location, "wb") as f:
                f.write(contents)

        # Call the ingestion process (ingest.py handles ChromaDB reset internally)
        num_files, num_chunks = ingest_pdfs()

        # Update in-memory count after successful indexing
        _chunk_count = num_chunks

        return {
            "message": "Indexing complete",
            "files_processed": num_files,
            "chunks_stored": num_chunks
        }
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[ERROR /index]: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        answer, sources_docs = get_answer(request.question, request.top_k)

        sources = []
        for doc in sources_docs:
            sources.append({
                "file_name": os.path.basename(doc.metadata.get('source', 'Unknown')),
                "page": doc.metadata.get('page', 0) + 1,
                "content": doc.page_content
            })

        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[ERROR /ask]: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Return stats using in-memory count. Never opens ChromaDB here to avoid file-lock conflicts."""
    return {
        "collection_name": "langchain",
        "chunk_count": _chunk_count,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "llama-3.1-8b-instant"
    }
