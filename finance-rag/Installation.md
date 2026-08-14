# 🛠️ Installation Guide — Finance RAG System

A complete step-by-step guide to set up the Finance RAG System on your machine from scratch.

---

## 📋 Table of Contents

1. [Install Python](#1--install-python)
2. [Verify Python & pip](#2--verify-python--pip)
3. [Clone the Project](#3--clone-the-project)
4. [Create a Virtual Environment](#4--create-a-virtual-environment)
5. [Install Dependencies](#5--install-dependencies)
6. [Get Your Groq API Key](#6--get-your-groq-api-key)
7. [Configure Environment Variables](#7--configure-environment-variables)
8. [Run the Application](#8--run-the-application)
9. [Troubleshooting](#9--troubleshooting)

---

## 1. 🐍 Install Python

You need **Python 3.10 or higher**. Download and install it from the official website.

### Windows

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click **"Download Python 3.x.x"** (latest stable version)
3. Run the installer
4. ⚠️ **IMPORTANT:** Check the box **"Add Python to PATH"** at the bottom of the installer before clicking "Install Now"

   ![Add to PATH](https://docs.python.org/3/_images/win_installer.png)

5. Click **"Install Now"**

### macOS

**Option A — Official Installer:**
1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download and run the macOS installer

**Option B — Using Homebrew (recommended):**

```bash
brew install python
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install python3 python3-pip
```

---

## 2. ✅ Verify Python & pip

Open a **new terminal** (Command Prompt, PowerShell, or Terminal) and run:

```bash
python --version
```

Expected output (version may vary):

```
Python 3.12.4
```

> **Note:** On some systems (especially Linux/macOS), you may need to use `python3` instead of `python`.

Now verify pip (Python's package manager — it comes bundled with Python):

```bash
pip --version
```

Expected output:

```
pip 24.x.x from ... (python 3.12)
```

### If pip is not installed

This is rare, but if `pip` is missing:

**Windows / macOS:**

```bash
python -m ensurepip --upgrade
```

**Linux:**

```bash
sudo apt install python3-pip        # Ubuntu/Debian
sudo dnf install python3-pip        # Fedora/RHEL
```

### Upgrade pip to the latest version

```bash
pip install --upgrade pip
```

---

## 3. 📥 Clone the Project

If you have **Git** installed:

```bash
git clone <your-repo-url>
cd finance-rag
```

If you **don't have Git**, install it first:

- **Windows:** Download from [https://git-scm.com/download/win](https://git-scm.com/download/win)
- **macOS:** `brew install git` or it comes with Xcode Command Line Tools (`xcode-select --install`)
- **Linux:** `sudo apt install git`

Alternatively, you can download the project as a ZIP file from your repository and extract it.

---

## 4. 🧪 Create a Virtual Environment

A virtual environment keeps this project's dependencies isolated from your system Python. **This is strongly recommended.**

### Create the virtual environment

```bash
python -m venv venv
```

This creates a `venv/` folder inside your project directory.

### Activate the virtual environment

**Windows (Command Prompt):**

```bash
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

> ⚠️ If you get a "running scripts is disabled" error in PowerShell, run this first:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### How to know it's working

After activation, your terminal prompt will show `(venv)` at the beginning:

```
(venv) PS E:\My_Projects\PCB_Assistance\finance-rag>
```

> **Tip:** You need to activate the virtual environment every time you open a new terminal to work on this project.

---

## 5. 📦 Install Dependencies

With your virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This installs the following packages:

| Package | Purpose |
|---|---|
| `streamlit` | Frontend web UI |
| `fastapi` | Backend REST API framework |
| `uvicorn` | ASGI server to run FastAPI |
| `langchain` | LLM orchestration framework |
| `langchain-groq` | Groq LLM integration |
| `langchain-huggingface` | HuggingFace embeddings integration |
| `langchain-community` | Community integrations (ChromaDB, PyPDF) |
| `chromadb` | Vector database for storing embeddings |
| `pypdf` | PDF parsing library |
| `sentence-transformers` | HuggingFace embedding models |
| `python-dotenv` | Load environment variables from `.env` file |
| `python-multipart` | File upload support for FastAPI |
| `requests` | HTTP client (Streamlit → FastAPI communication) |

> **Note:** The first run may take a few minutes as it downloads model files (~80 MB for `all-MiniLM-L6-v2`).

### Verify installation

```bash
pip list
```

You should see all the packages listed above in the output.

---

## 6. 🔑 Get Your Groq API Key

This project uses **Groq** as the LLM provider. You need a free API key.

1. Go to [https://console.groq.com/](https://console.groq.com/)
2. Sign up for a free account (or log in if you already have one)
3. Navigate to **API Keys** → [https://console.groq.com/keys](https://console.groq.com/keys)
4. Click **"Create API Key"**
5. Give it a name (e.g., `finance-rag`) and copy the key

> ⚠️ **Save the key immediately** — Groq only shows it once. If you lose it, you'll need to create a new one.

---

## 7. ⚙️ Configure Environment Variables

Create a `.env` file from the provided template:

**Windows (Command Prompt):**

```bash
copy .env.example .env
```

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

Now open the `.env` file in any text editor and add your Groq API key:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> 🔒 **Security:** The `.env` file is listed in `.gitignore` and will NOT be committed to Git. Never share your API key publicly.

---

## 8. 🚀 Run the Application

This project requires **two terminals running simultaneously** — one for the backend API and one for the frontend UI.

### Terminal 1 — Start the FastAPI Backend

```bash
uvicorn api:app --reload
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

✅ Backend is now running at: **http://localhost:8000**

📖 API docs available at: **http://localhost:8000/docs**

### Terminal 2 — Start the Streamlit Frontend

Open a **new terminal**, navigate to the project folder, activate the virtual environment again, and run:

**Windows:**

```bash
cd E:\My_Projects\PCB_Assistance\finance-rag
venv\Scripts\activate
streamlit run app.py
```

**macOS / Linux:**

```bash
cd finance-rag
source venv/bin/activate
streamlit run app.py
```

You should see:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

✅ Frontend is now running at: **http://localhost:8501**

### Quick Test

1. Open **http://localhost:8501** in your browser
2. Upload a PDF in the sidebar → click **"Save Uploaded Files"**
3. Click **"Index Documents"** → wait for success message
4. Type a question in the chat box and press Enter

---

## 9. 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `python` command not found | Use `python3` instead, or re-install Python with **"Add to PATH"** checked |
| `pip` command not found | Run `python -m ensurepip --upgrade` |
| PowerShell script execution error | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ModuleNotFoundError` | Make sure your virtual environment is activated (`(venv)` in prompt) |
| `AuthenticationError` from Groq | Double-check `GROQ_API_KEY` in your `.env` file |
| `Backend API is not running` in UI | Start the FastAPI server first in Terminal 1 |
| `No PDFs found to index` | Click **"Save Uploaded Files"** before indexing |
| ChromaDB file lock error (Windows) | Close both terminals, restart the servers |
| `model_decommissioned` error from Groq | The model ID has changed — update `model_name` in `rag.py` to a supported model (see [Groq Models](https://console.groq.com/docs/models)) |
| Slow first run | Normal — the embedding model (~80 MB) downloads on first use |

---

## 📌 Quick Reference Commands

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend (Terminal 1)
uvicorn api:app --reload

# Start frontend (Terminal 2)
streamlit run app.py

# Deactivate virtual environment
deactivate
```

---

> 💡 **Need help?** Open an issue on the repository or refer to the [README.md](README.md) for architecture details and API reference.
