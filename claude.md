# PCB Assistance — Project Guide

## Project Overview

A **RAG-based (Retrieval-Augmented Generation) PCB design assistant** that answers questions about PCB design using a pre-processed knowledge base extracted from a 1,343-page Sierra Circuits PDF. The backend uses **BM25 text retrieval** to find relevant chunks and **Groq's LLaMA 3.3 70B** to generate detailed answers. The frontend is a single-page chat UI served by FastAPI itself.

## Architecture

```
┌───────────────────────────────────────────────────┐
│                   User Browser                    │
│            FrontEnd/index.html                    │
│   (Vanilla HTML/CSS/JS + marked.js for markdown)  │
└────────────────────┬──────────────────────────────┘
                     │  POST /chat  { text: "..." }
                     │  GET  /      (serves index.html)
                     ▼
┌───────────────────────────────────────────────────┐
│              FastAPI Backend (app.py)              │
│                                                   │
│  1. Load chunks.json at startup                   │
│  2. Build BM25 index in-memory                    │
│  3. On /chat → BM25 search → top-10 chunks       │
│  4. Send chunks + question to Groq LLM           │
│  5. Return { answer: "..." }                      │
│                                                   │
│  Also serves FrontEnd/ as static files            │
└───────────────────────────────────────────────────┘
```

## Data Pipeline Flow

1. **`build_chunks.py`** (offline, run once):
   - Opens `ilovepdf_merged.pdf` (root directory) with PyMuPDF
   - Splits each page's text into overlapping 200-word chunks (stride of 150 words)
   - Saves all chunks with page numbers to `Backend/chunks.json`

2. **`app.py`** (runtime):
   - Loads `chunks.json` on startup
   - Builds a **BM25** index (custom implementation — no external library)
   - The BM25 class tokenizes text, removes stopwords, computes TF/IDF/document-length normalization
   - On each query: retrieves top-10 relevant chunks (score > 0.1)
   - Passes sorted chunks (by page number) as context to Groq's `llama-3.3-70b-versatile` model
   - The system prompt instructs the LLM to be a detailed PCB technical assistant with page references

## File Structure

```
PCB_Assistance/
├── Backend/
│   ├── .env                 # GROQ_API_KEY (gitignored)
│   ├── __init__.py          # Makes Backend a Python package
│   ├── app.py               # FastAPI server + BM25 + Groq integration
│   ├── build_chunks.py      # PDF → chunks.json preprocessing script
│   ├── chunks.json          # Pre-built text chunks (~1.8MB)
│   ├── faiss.index          # FAISS index file (currently unused by app.py)
│   └── requirements.txt     # Python dependencies
├── FrontEnd/
│   └── index.html           # Complete chat UI (HTML + CSS + JS in one file)
├── DockerFile               # Docker deployment config (port 10000)
├── .gitignore               # Ignores .env, __pycache__, PDFs
└── ilovepdf_merged.pdf      # Source PDF (~163MB, gitignored)
```

## Key Technical Details

### BM25 Implementation (app.py, lines 30-75)
- Custom pure-Python BM25 with parameters `k1=1.5`, `b=0.75`
- Tokenization: lowercase, removes common English stopwords, filters words ≤ 2 chars
- Returns top-10 chunks with score > 0.1

### Frontend (FrontEnd/index.html)
- Single HTML file with embedded CSS and JS
- Dark theme with gold accent (`#d4a843`) and PCB circuit-line grid background
- Uses `marked.js` CDN to render markdown in AI responses
- Suggestion cards trigger pre-defined questions
- Textarea auto-grows; Enter sends, Shift+Enter for newline
- Fetches `/chat` endpoint; shows loading state during API call

### API Endpoint
- **`POST /chat`** — Body: `{ "text": "user question" }` → Response: `{ "answer": "..." }`
- **`GET /`** — Serves `FrontEnd/index.html`
- **`/static/*`** — Serves files from `FrontEnd/` directory

## Running the Project

```bash
# 1. Install dependencies
cd Backend
pip install -r requirements.txt

# 2. Set up environment
# Create Backend/.env with: GROQ_API_KEY=your_key_here

# 3. (Optional) Rebuild chunks from PDF
python build_chunks.py

# 4. Start the server
uvicorn Backend.app:app --reload --port 8000
# Run from project root (PCB_Assistance/), NOT from Backend/
```

## Important Notes

- **Do NOT change the code flow.** The BM25 → Groq pipeline is intentional. Do not replace BM25 with FAISS or embeddings.
- The `faiss.index` file exists but is **not used** in the current app.py. Do not wire it in.
- The frontend fetches from relative `/chat` path — it expects to be served by the same FastAPI server.
- `chunks.json` is pre-built and checked in. No need to re-run `build_chunks.py` unless the PDF changes.
- Docker is configured to run on port `10000` using `Backend.app:app` module path.
- The `.env` file is gitignored — each developer must create their own with a valid Groq API key.