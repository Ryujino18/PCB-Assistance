import os
import json
import math
import fitz  # PyMuPDF
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
PDF_FILE = os.path.join(BASE_DIR, "..", "ilovepdf_merged.pdf")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "FrontEnd")

# ── Extract text from PDF with small overlapping chunks ──
def extract_chunks(pdf_path, chunk_size=200, overlap=50):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue
        words = text.split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page_num + 1
                })
            i += chunk_size - overlap  # overlap for continuity
    return chunks

# ── BM25 Search ──
class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)
        self.avgdl = 0
        self.df = defaultdict(int)
        self.tf = []
        self._build()

    def _tokenize(self, text):
        # remove common words for better matching
        stopwords = {"the","a","an","is","in","it","of","and","or","to","for","on","with","that","this","are","was","be","as","at","by","from","have","has"}
        words = text.lower().split()
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _build(self):
        total_len = 0
        for chunk in self.chunks:
            words = self._tokenize(chunk["text"])
            total_len += len(words)
            freq = defaultdict(int)
            for w in words:
                freq[w] += 1
            self.tf.append(freq)
            for w in set(words):
                self.df[w] += 1
        self.avgdl = total_len / self.N if self.N > 0 else 1

    def score(self, query, idx):
        words = self._tokenize(query)
        doc_len = sum(self.tf[idx].values())
        score = 0.0
        for w in words:
            if w not in self.tf[idx]:
                continue
            tf = self.tf[idx][w]
            df = self.df.get(w, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            norm_tf = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            score += idf * norm_tf
        return score

    def search(self, query, top_k=10, min_score=0.1):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        seen_pages = set()
        for i, s in scores:
            if s < min_score:
                break
            results.append(self.chunks[i])
            seen_pages.add(self.chunks[i]["page"])
            if len(results) >= top_k:
                break
        return results

# ── Ask Groq with all relevant context ──
def ask(query, context_chunks):
    if not context_chunks:
        return "I couldn't find relevant information in the book for your question."

    # Sort by page number for logical reading order
    context_chunks_sorted = sorted(context_chunks, key=lambda x: x["page"])
    
    # Build context with page references
    context_parts = []
    for c in context_chunks_sorted:
        context_parts.append(f"[Page {c['page']}]: {c['text']}")
    
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2048,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a detailed PCB technical assistant. "
                    "Your job is to give COMPLETE and THOROUGH answers based on the provided book content. "
                    "Include ALL relevant details, definitions, specifications, and explanations found in the content. "
                    "Mention page numbers when referencing specific information. "
                    "Do not summarize or skip any relevant detail. "
                    "If the content covers multiple aspects of the question, explain all of them."
                )
            },
            {
                "role": "user",
                "content": f"Book content:\n{context}\n\nQuestion: {query}\n\nProvide a complete and detailed answer using all relevant information from the book content above."
            }
        ]
    )
    return response.choices[0].message.content

# ── Startup ──
print("Initializing...")
if os.path.exists(CHUNKS_FILE):
    print("Saved chunks found! Loading...")
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)
else:
    print("No chunks found. Reading PDF...")
    chunks = extract_chunks(PDF_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks!")

print(f"Building BM25 index over {len(chunks)} chunks...")
bm25 = BM25(chunks)
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
    results = bm25.search(query.text, top_k=10)
    answer = ask(query.text, results)
    return {"answer": answer}