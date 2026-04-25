import os
import json
import faiss
import numpy as np
import fitz  # PyMuPDF
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Paths work both locally and on Render ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
INDEX_FILE = os.path.join(BASE_DIR, "faiss.index")
PDF_FILE = os.path.join(BASE_DIR, "..", "ilovepdf_merged.pdf")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "FrontEnd")

# ── Simple embedding using hash trick (no heavy model needed) ──
def simple_embed(text, dim=384):
    vec = np.zeros(dim, dtype="float32")
    words = text.lower().split()
    for word in words:
        idx = hash(word) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

# ── Extract text from PDF ──
def extract_chunks(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text().strip()
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk:
                chunks.append(chunk)
    return chunks

# ── Build FAISS index ──
def build_index(chunks):
    print("Building index from PDF...")
    embeddings = np.stack([simple_embed(c) for c in chunks])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    print("Index built and saved!")
    return index, chunks

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)
    return index, chunks

# ── Search ──
def search(query, index, chunks, top_k=4):
    query_vec = simple_embed(query).reshape(1, -1)
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ── Ask Groq ──
def ask(query, context_chunks):
    context = "\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided book content."},
            {"role": "user", "content": f"Book content:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

# ── Startup ──
print("Initializing...")
if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    print("Saved index found! Loading...")
    index, chunks = load_index()
    print("Ready!")
else:
    print("No index found. Reading PDF...")
    chunks = extract_chunks(PDF_FILE)
    index, chunks = build_index(chunks)
    print("Ready!")

# ── Serve frontend ──
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ── API endpoint ──
class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    results = search(query.text, index, chunks)
    answer = ask(query.text, results)
    return {"answer": answer}