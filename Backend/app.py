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
embedder = SentenceTransformer("all-MiniLM-L6-v2")

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "faiss.index"

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

def build_index(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    return index, chunks

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)
    return index, chunks

def search(query, index, chunks, top_k=4):
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask(query, context_chunks):
    context = "\n\n".join(context_chunks)
    
    # SYSTEM PROMPT FOR CLAUDE-STYLE FORMATTING
    system_prompt = (
        "You are a professional PCB Assistant. Answer based only on the provided book content. "
        "Structure your response strictly by following these rules:\n"
        "1. Use **Bold Headings** to separate different parts of your answer.\n"
        "2. Use bullet points (*) or numbered lists for steps and details.\n"
        "3. Use arrows (->) to show workflows or logical connections.\n"
        "4. Use **bold text** for important technical terms.\n"
        "5. Ensure the answer is organized, professional, and easy to read."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Book content:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

# Startup Logic
if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    index, chunks = load_index()
else:
    pdf_path = "ilovepdf_merged.pdf" if os.path.exists("ilovepdf_merged.pdf") else "../ilovepdf_merged.pdf"
    chunks = extract_chunks(pdf_path)
    index, chunks = build_index(chunks)

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    results = search(query.text, index, chunks)
    answer = ask(query.text, results)
    return {"answer": answer}