import os
import json
import faiss
import numpy as np
import fitz  # PyMuPDF
from groq import Groq
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Free, runs locally

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "faiss.index"

# ── Step 1: Extract text from PDF and split into chunks ──
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

# ── Step 2: Build or load FAISS index ──
def build_index(chunks):
    print("Building index from PDF... this may take a few minutes.")
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
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

# ── Step 3: Search for relevant chunks ──
def search(query, index, chunks, top_k=4):
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ── Step 4: Ask Groq with context ──
def ask(query, context_chunks):
    context = "\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Updated model
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided book content."},
            {"role": "user", "content": f"Book content:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

# ── Startup: load or build index ──
print("Initializing...")
if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    print("Saved index found! Loading...")
    index, chunks = load_index()
    print("Ready!")
else:
    print("No index found. Reading PDF...")
    chunks = extract_chunks("../ilovepdf_merged.pdf")
    index, chunks = build_index(chunks)
    print("Ready!")

# ── API endpoint ──
class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    results = search(query.text, index, chunks)
    answer = ask(query.text, results)
    return {"answer": answer}