import os
import json
import math
import fitz  # PyMuPDF
import requests
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
PDF_FILE = os.path.join(BASE_DIR, "ilovepdf_merged.pdf")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "FrontEnd")

GDRIVE_FILE_ID = "1PVNpBEhBbHxSEvgRlyLmMlc0u_zUEqnr"

# ── Download PDF from Google Drive if not present ──
def download_pdf():
    if os.path.exists(PDF_FILE):
        print("PDF already exists, skipping download.")
        return
    print("Downloading PDF from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    session = requests.Session()
    response = session.get(url, stream=True)
    # Handle large file warning page
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm={value}"
            response = session.get(url, stream=True)
            break
    with open(PDF_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print(f"PDF downloaded successfully!")

# ── Extract text from PDF with overlapping chunks ──
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
            i += chunk_size - overlap
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
        for i, s in scores:
            if s < min_score:
                break
            results.append(self.chunks[i])
            if len(results) >= top_k:
                break
        return results

# ── Ask Groq ──
def ask(query, context_chunks):
    if not context_chunks:
        return "I couldn't find relevant information in the book for your question."
    context_chunks_sorted = sorted(context_chunks, key=lambda x: x["page"])
    context_parts = [f"[Page {c['page']}]: {c['text']}" for c in context_chunks_sorted]
    context = "\n\n".join(context_parts)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2048,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a detailed PCB technical assistant. "
                    "Give COMPLETE and THOROUGH answers based on the provided book content. "
                    "Include ALL relevant details, definitions, specifications, and explanations. "
                    "Mention page numbers when referencing specific information. "
                    "Do not summarize or skip any relevant detail."
                )
            },
            {
                "role": "user",
                "content": f"Book content:\n{context}\n\nQuestion: {query}\n\nProvide a complete and detailed answer."
            }
        ]
    )
    return response.choices[0].message.content

# ── Startup ──
print("Initializing...")
download_pdf()

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