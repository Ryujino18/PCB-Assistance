# 🔌 PCB Assistance — RAG-Based Technical PCB Assistant

A **Retrieval-Augmented Generation (RAG)** application designed to answer complex technical questions on Printed Circuit Board (PCB) design, manufacturing, stackups, signal integrity, and layout guidelines.

The system retrieves context from a pre-processed knowledge base (extracted from a 1,343-page Sierra Circuits PCB Design reference book) using a pure-Python **BM25 text ranking engine**, and generates page-referenced bulleted answers using **Groq's LLaMA 3.3 70B** model.

---

## 🌟 Key Features

- **Strict Knowledge-Grounded Answering**: Answers strictly based on book content with explicit page citations **(Page X)** for easy verification.
- **Fast BM25 Keyword Search**: In-memory BM25 retrieval (`k1=1.5`, `b=0.75`) providing instant token matching across 2,200+ text chunks without heavy vector database dependencies.
- **Powered by Groq LLM**: High-speed inference using `llama-3.3-70b-versatile` via the Groq API.
- **Custom PCB Circuit UI**: Sleek, dark-mode single-page interface with gold circuit accents (`#d4a843`), interactive suggestion cards, dynamic markdown parsing (`marked.js`), and responsive layout.
- **Unified FastAPI Backend**: Single service serving both the `/chat` API endpoint and static frontend files (`/`).
- **Docker Ready**: Pre-configured Dockerfile for containerized deployment.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       User Browser                          │
│                   FrontEnd/index.html                       │
│    (Vanilla HTML/CSS/JS + marked.js Markdown Parser)        │
└──────────────────────────────┬──────────────────────────────┘
                               │  POST /chat  { "text": "..." }
                               │  GET  /      (Serves index.html)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Backend (Backend/app.py)             │
│                                                             │
│  1. Startup: Load Backend/chunks.json into memory           │
│  2. Build: In-memory BM25 term index                        │
│  3. /chat Request:                                          │
│     ├── Tokenize query & score text chunks                  │
│     ├── Select top-10 chunks (BM25 score > 0.1)             │
│     ├── Pass sorted chunks + user question to Groq API      │
│     └── Return structured bulleted answer with page references│
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
PCB_Assistance/
├── Backend/
│   ├── .env                 # Environment variables (GROQ_API_KEY)
│   ├── __init__.py          # Python package initializer
│   ├── app.py               # FastAPI server, BM25 algorithm & Groq integration
│   ├── build_chunks.py      # PDF text extractor & chunk generator script
│   ├── chunks.json          # Pre-built dataset (~2,223 text chunks, ~1.8 MB)
│   ├── faiss.index          # FAISS index (archived / reserved)
│   └── requirements.txt     # Python dependencies
├── FrontEnd/
│   └── index.html           # Single-page frontend application
├── DockerFile               # Container setup configuration
├── README.md                # Project documentation & execution guide
├── claude.md                # Technical guide & architectural specs
├── .gitignore               # Ignored files (.env, __pycache__, PDFs)
└── ilovepdf_merged.pdf      # Source reference PDF (1,343 pages, optional/gitignored)
```

---

## ⚙️ Prerequisites

Before running the application, make sure you have:

1. **Python 3.9+** installed on your system.
2. A **Groq API Key**:
   - Get your API key from [Groq Console](https://console.groq.com/).

---

## 🚀 How to Run the Program

Follow these step-by-step instructions to get the application running on your local machine.

### Step 1: Open Terminal & Navigate to Project Root

Ensure you are in the project root folder `PCB_Assistance`:

```bash
cd e:\My_Projects\PCB_Assistance
```

---

### Step 2: Install Required Dependencies

Navigate into the `Backend` directory or run `pip` directly pointing to `Backend/requirements.txt`:

```bash
pip install -r Backend/requirements.txt
```

*(Dependencies include: `fastapi`, `uvicorn`, `groq`, `pymupdf`, `numpy`, `pydantic`, `python-dotenv`, `aiofiles`, `requests`)*

---

### Step 3: Configure Environment Variables

Create a file named `.env` inside the `Backend/` directory:

**`Backend/.env`**
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

> ⚠️ **Note**: Replace `your_actual_groq_api_key_here` with your valid Groq API key string starting with `gsk_...`.

---

### Step 4: (Optional) Preprocess PDF Chunks

The repository already includes `Backend/chunks.json` (2,223 pre-extracted chunks). You only need to run this step if you update or replace `ilovepdf_merged.pdf` in the root folder:

```bash
cd Backend
python build_chunks.py
cd ..
```

---

### Step 5: Start the FastAPI Application Server

> 📌 **IMPORTANT**: Always run the `uvicorn` command **from the project root directory** (`PCB_Assistance`), **NOT** from inside `Backend/`.

Run the following command in your terminal:

```bash
uvicorn Backend.app:app --reload --port 8000
```

Upon starting, you will see server output similar to:
```text
Initializing...
Loaded 2223 chunks!
Ready!
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

### Step 6: Open the Application in Your Web Browser

Open your browser and navigate to:

👉 **[http://localhost:8000](http://localhost:8000)** or **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You will be presented with the PCB Assistant chat interface!

---

## 🐳 Running with Docker (Alternative)

If you prefer using Docker:

1. **Build the Docker Image**:
   ```bash
   docker build -t pcb-assistance .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -d -p 10000:10000 --env-file Backend/.env --name pcb_app pcb-assistance
   ```

3. **Access the Containerized App**:
   Open **[http://localhost:10000](http://localhost:10000)** in your browser.

---

## 🛠️ API Reference

- **`GET /`**
  Serves the `FrontEnd/index.html` single-page web app.

- **`POST /chat`**
  Processes user queries through BM25 search & Groq LLM inference.
  - **Request Body**:
    ```json
    {
      "text": "What is the recommended clearance for high voltage traces?"
    }
    ```
  - **Response**:
    ```json
    {
      "answer": "* High voltage trace clearance recommendations...\n* Minimum spacing guidelines **(Page 142)**"
    }
    ```

- **Interactive API Documentation (Swagger UI)**:
  Available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🔍 Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `groq.APIConnectionError` or `GroqError` | Invalid or missing `GROQ_API_KEY` | Ensure `Backend/.env` exists and contains a valid `GROQ_API_KEY=gsk_...`. |
| `ModuleNotFoundError: No module named 'Backend'` | Uvicorn launched inside `Backend/` folder | Run `cd ..` to return to the project root and execute `uvicorn Backend.app:app --reload --port 8000`. |
| `FileNotFoundError: chunks.json` | `chunks.json` missing | Run `python Backend/build_chunks.py` to regenerate the chunk index from the PDF. |
