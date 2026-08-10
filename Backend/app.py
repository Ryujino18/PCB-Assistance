import os
import json
import math
from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from collections import defaultdict
import pymupdf as fitz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
CHUNKS_FILE = os.path.join(BASE_DIR, "chunks.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "FrontEnd")

class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)
        self.df = defaultdict(int)
        self.tf = []
        self.avgdl = 0
        self._build()

    def _tokenize(self, text):
        stopwords = {"the","a","an","is","in","it","of","and","or","to","for","on","with","that","this","are","was","be","as","at","by","from","have","has"}
        return [w for w in text.lower().split() if w not in stopwords and len(w) > 2]

    def _build(self):
        total = 0
        for chunk in self.chunks:
            words = self._tokenize(chunk["text"])
            total += len(words)
            freq = defaultdict(int)
            for w in words:
                freq[w] += 1
            self.tf.append(freq)
            for w in set(words):
                self.df[w] += 1
        self.avgdl = total / self.N if self.N else 1

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

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.chunks[i] for i, s in scores[:top_k] if s > 0.1]

def ask(query, context_chunks):
    if not context_chunks:
        return "I couldn't find relevant information in the book for your question."
    context_chunks_sorted = sorted(context_chunks, key=lambda x: x["page"])
    context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in context_chunks_sorted])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2048,
        messages=[
            {"role": "system", "content": "You are a precise PCB technical assistant. Answer ONLY from the provided book content. Follow these rules strictly:\n1. Your entire response MUST be a strict bulleted list format. Do not use conversational paragraphs.\n2. Each point should be concise, precise, and directly answer the user's question.\n3. Always include the **(Page X)** reference at the exact end of each bullet point.\n4. Do NOT add any outside information.\n5. Use bold for key terms.\n6. Do not include introductory or concluding conversational filler."},
            {"role": "user", "content": f"Book content:\n{context}\n\nQuestion: {query}\n\nProvide a precise, point-wise answer sourced strictly from the book content above."}
        ]
    )
    return response.choices[0].message.content

print("Initializing...")
with open(CHUNKS_FILE, "r") as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks!")
bm25 = BM25(chunks)
print("Ready!")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    results = bm25.search(query.text)
    answer = ask(query.text, results)
    return {"answer": answer}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global chunks, bm25
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    new_chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if not text:
            continue
        words = text.split()
        for i in range(0, len(words), 150):
            chunk_text = " ".join(words[i:i+200])
            if chunk_text:
                new_chunks.append({"text": chunk_text, "page": page_num + 1})
    doc.close()
    if not new_chunks:
        raise HTTPException(status_code=422, detail="No readable text found in the uploaded PDF.")
    chunks = new_chunks
    bm25 = BM25(chunks)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)
    return {"message": f"Successfully processed '{file.filename}' with {len(chunks)} chunks."}