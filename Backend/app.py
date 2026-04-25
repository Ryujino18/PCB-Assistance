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

# ── Download PDF from Google Drive (handles large file confirmation) ──
def download_pdf():
    if os.path.exists(PDF_FILE) and os.path.getsize(PDF_FILE) > 1000000:
        print(f"PDF already exists ({os.path.getsize(PDF_FILE)} bytes), skipping download.")
        return True

    print("Downloading PDF from Google Drive...")
    session = requests.Session()

    # Step 1: Get the confirmation token for large files
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    response = session.get(url, stream=True)

    # Find confirmation token
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    # Also check in response content for newer Google Drive format
    if not confirm_token:
        import re
        content = response.text[:5000]
        match = re.search(r'confirm=([0-9A-Za-z_]+)', content)
        if match:
            confirm_token = match.group(1)

    # Step 2: Download with confirmation
    if confirm_token:
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm={confirm_token}"
        response = session.get(url, stream=True)

    # Step 3: Save file
    total = 0
    with open(PDF_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size = os.path.getsize(PDF_FILE)
    print(f"Downloaded {size} bytes.")

    if size < 1000000:
        print("ERROR: Downloaded file is too small — Google Drive may have blocked it!")
        os.remove(PDF_FILE)
        return False

    print("PDF downloaded successfully!")
    return True

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
                chunks.append({"text": chunk, "page": page_num + 1})
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
pdf_ok = download_pdf()

if not pdf_ok:
    print("FATAL: Could not download PDF. Answers will not work.")
    chunks = []
elif os.path.exists(CHUNKS_FILE):
    print("Saved chunks found! Loading...")
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)
    # Validate format
    if chunks and isinstance(chunks[0], str):
        print("Old format detected, rebuilding chunks...")
        os.remove(CHUNKS_FILE)
        chunks = extract_chunks(PDF_FILE)
        with open(CHUNKS_FILE, "w") as f:
            json.dump(chunks, f)
else:
    print("No chunks found. Reading PDF...")
    chunks = extract_chunks(PDF_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks!")

if chunks:
    print(f"Building BM25 index over {len(chunks)} chunks...")
    bm25 = BM25(chunks)
    print("Ready!")
else:
    bm25 = None
    print("WARNING: No chunks loaded. Check PDF download.")

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
    if not bm25:
        return {"answer": "System is still initializing or PDF failed to load. Please try again in a moment."}
    results = bm25.search(query.text, top_k=10)
    answer = ask(query.text, results)
    return {"answer": answer}