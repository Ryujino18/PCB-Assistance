# 📈 Finance RAG — Quarterly Financial Reports Assistant

A **Retrieval-Augmented Generation (RAG)** application that lets you upload quarterly financial report PDFs, index them into a persistent vector database, and query them using natural language — all powered by **Groq Llama 3**.

Every answer is strictly grounded in the uploaded documents and includes **source citations** (file name + page number). If the answer isn't in the documents, the system says so honestly — no hallucinations.

---

## 🌟 Key Features

| Feature | Details |
|---|---|
| 📄 **PDF Upload & Indexing** | Upload multiple PDFs via the UI; documents are chunked and embedded automatically |
| 🧠 **HuggingFace Embeddings** | Uses `all-MiniLM-L6-v2` for high-quality semantic search |
| 💬 **Groq Answering** | Powered by `llama3-8b-8192` with temperature 0 for deterministic, factual answers |
| 🗃️ **Persistent Vector Store** | ChromaDB persists embeddings to disk — survives app restarts |
| 📌 **Source Citations** | Every answer cites the exact file name and page number |
| 🚫 **Honest Refusal** | Plainly says "not available" if the answer isn't in the documents |
| 🌐 **Decoupled Architecture** | FastAPI backend (`api.py`) + Streamlit frontend (`app.py`) communicate over HTTP |

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐       HTTP        ┌──────────────────────────────────┐
│        Streamlit Frontend        │ ◄───────────────► │        FastAPI Backend            │
│           app.py                 │  POST /index       │           api.py                 │
│                                  │  POST /ask         │                                  │
│  - Upload PDFs                   │  GET  /stats       │  - /index  → ingest.py           │
│  - Trigger indexing              │                    │  - /ask    → rag.py              │
│  - Chat interface                │                    │  - /stats  → chunk metadata      │
└─────────────────────────────────┘                    └──────────────────────────────────┘
                                                                      │
                                                          ┌───────────┴──────────┐
                                                          │                      │
                                                    ┌─────▼─────┐         ┌──────▼──────┐
                                                    │ ingest.py  │         │   rag.py    │
                                                    │            │         │             │
                                                    │ PyPDFLoader│         │ ChromaDB    │
                                                    │ Chunking   │         │ Retrieval   │
                                                    │ HuggingFace│         │ Groq Llama 3│
                                                    │ Embeddings │         │ Answer Gen  │
                                                    │ ChromaDB   │         │             │
                                                    └─────┬──────┘         └──────┬──────┘
                                                          │                       │
                                                          └──────────┬────────────┘
                                                                     │
                                                             ┌───────▼───────┐
                                                             │  chroma_db/   │
                                                             │ (Persistent   │
                                                             │  Vector Store)│
                                                             └───────────────┘
```

---

## 📁 Project Structure

```
finance-rag/
├── app.py              # Streamlit frontend — upload, index, and chat UI
├── api.py              # FastAPI backend — /index, /ask, /stats endpoints
├── ingest.py           # PDF loader → chunker → embedder → ChromaDB writer
├── rag.py              # ChromaDB retriever → GPT-4o answering chain
├── data/               # Uploaded PDF files are stored here
├── chroma_db/          # Persisted ChromaDB vector embeddings (auto-created)
├── requirements.txt    # Python dependencies
├── .env                # Your API keys (never commit this)
├── .env.example        # Template for environment variables
├── .gitignore          # Excludes .env, chroma_db/, data/ from git
└── README.md           # This file
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq `llama3-8b-8192` |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | ChromaDB (persistent, local) |
| **PDF Parsing** | `pypdf` via LangChain's `PyPDFLoader` |
| **Chunking** | `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend UI** | Streamlit |
| **Orchestration** | LangChain |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A **Groq API key** with access to `llama3-8b-8192`

### Step 1: Clone & Navigate

```bash
cd finance-rag
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Copy the example file and add your Groq key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Important:** Never commit your `.env` file. It is already listed in `.gitignore`.

---

## ▶️ Running the Application

This project uses a **two-process architecture**. You need to run both the backend and the frontend simultaneously — in two separate terminals.

### Terminal 1 — Start the FastAPI Backend

```bash
uvicorn api:app --reload
```

The API will be available at: **http://localhost:8000**

You can also explore the auto-generated API docs at: **http://localhost:8000/docs**

### Terminal 2 — Start the Streamlit Frontend

```bash
streamlit run app.py
```

The UI will open automatically at: **http://localhost:8501**

> **Note:** The Streamlit app checks if the backend is reachable on startup. If you see a red warning message in the sidebar, make sure the FastAPI server is running first.

---

## 📖 How to Use

1. **Upload** — In the sidebar, click **"Upload PDF files"** and select one or more quarterly report PDFs. Click **"Save Uploaded Files"**.

2. **Index** — Click **"2. Index Documents"** in the sidebar. Wait for the success message showing how many files and chunks were processed. This step embeds your documents into ChromaDB.

3. **Ask** — Type your question in the chat box at the bottom (e.g., *"What was the revenue in Q1 FY26?"*) and press Enter.

4. **Review Sources** — Expand the **"View Sources"** section below any answer to see the exact file name and page number the answer was drawn from.

> **Tip:** For best results, name your PDF files descriptively (e.g., `Infosys_Q1_FY26.pdf`) — the filename is used as a source label in the context passed to the LLM.

---

## 🔌 API Reference

The FastAPI backend exposes three endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/index` | Accepts uploaded PDF files, runs ingestion, and stores embeddings |
| `POST` | `/ask` | Accepts a `question` and optional `top_k`, returns answer + sources |
| `GET` | `/stats` | Returns collection name, chunk count, and model info |

### `/ask` Request Body

```json
{
  "question": "What was the net profit for Q2 FY25?",
  "top_k": 4
}
```

### `/ask` Response

```json
{
  "answer": "The net profit for Q2 FY25 was ₹6,506 crore.",
  "sources": [
    {
      "file_name": "Infosys_Q2_FY25.pdf",
      "page": 12,
      "content": "..."
    }
  ]
}
```

---

## 🗂️ How Ingestion Works

1. **Load** — `PyPDFLoader` reads each PDF page-by-page, preserving page number metadata.
2. **Chunk** — `RecursiveCharacterTextSplitter` splits pages into 1000-character chunks with 150-character overlaps to preserve context across boundaries.
3. **Label** — Each chunk is prepended with its source filename to help the LLM identify the source during retrieval.
4. **Embed** — HuggingFace `all-MiniLM-L6-v2` converts each chunk into a vector.
5. **Store** — ChromaDB persists all vectors to the `chroma_db/` directory. Re-indexing wipes the old collection first to avoid duplicates.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `Backend API is not running` warning in UI | Start the FastAPI server: `uvicorn api:app --reload` |
| `AuthenticationError` from Groq | Check that `GROQ_API_KEY` is correctly set in your `.env` file |
| `No PDFs found to index` | Click **"Save Uploaded Files"** before clicking **"Index Documents"** |
| Answers seem outdated after new uploads | Re-click **"Index Documents"** to rebuild the vector store |
| ChromaDB file lock error on Windows | Restart both processes; Windows does not allow multiple connections to the same ChromaDB files |

---

## 📄 License

This project is for educational and research purposes.
