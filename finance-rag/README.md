# 📈 Finance RAG — Quarterly Financial Reports Assistant

A **Retrieval-Augmented Generation (RAG)** application built to satisfy specific technical assignment requirements. This system allows users to upload PDF reports, index them into a vector database, and query them using an LLM.

The system ensures **strict knowledge-grounded answering** by returning information solely from the uploaded PDFs and clearly citing the **file name and page number** for every answer. It includes a built-in "Honest Refusal" mechanism if the answer cannot be found in the context.

---

## 🌟 Key Features

- **Upload & Index via UI**: Users can upload multiple PDFs directly from the frontend.
- **Advanced Chunking**: Documents are processed using `RecursiveCharacterTextSplitter` with 1000 character chunks and 150 character overlaps.
- **Local Embeddings**: Uses HuggingFace's `all-MiniLM-L6-v2` embedding model (completely free, replacing OpenAI's paid `text-embedding-3-small`).
- **Persistent Vector Database**: Stores vector embeddings in `ChromaDB` inside the `chroma_db/` folder, ensuring indexed documents remain searchable after application restarts.
- **Powered by Groq LLM**: Uses `llama-3.3-70b-versatile` via the Groq API (as a free alternative to GPT-4o).
- **Source Citation**: Every answer includes the exact file name and page number as required.
- **Interactive Interface**: Sleek Streamlit user interface (`app.py`).

---

## 📁 Repository Structure

```
finance-rag/
├── app.py              # Streamlit interface (Upload, Index, Chat)
├── ingest.py           # Loads PDFs, chunks text, embeds, stores in ChromaDB
├── rag.py              # Retrieves context from ChromaDB + Prompts Groq LLM
├── data/               # Directory where uploaded PDFs are stored
├── chroma_db/          # Persisted Chroma vector database
├── requirements.txt    # Python dependencies
├── .env                # API Keys (GROQ_API_KEY)
├── .env.example        # Example environment template
├── .gitignore          # Git ignore rules
└── README.md           # This documentation file
```

---

## 🚀 How to Run the Application

Follow these steps to launch the `finance-rag` Streamlit app on your local machine:

### Step 1: Open Terminal
Open a new terminal or PowerShell window and navigate into the `finance-rag` directory:
```bash
cd finance-rag
```

### Step 2: Install Dependencies
Ensure you have installed all the necessary Python packages:
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Make sure your `.env` file exists in the `finance-rag` folder and contains your Groq API key:
```env
GROQ_API_KEY=your_key_here
```

### Step 4: Run the Streamlit App
Start the Streamlit development server by running:
```bash
streamlit run app.py
```

### Step 5: Open the Application
Your browser should automatically open the app. If it doesn't, navigate to:

👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📖 How to Use the App

1. **Upload**: Use the sidebar to upload one or more PDF files. Click **Save Uploaded Files**.
2. **Index**: Click the **Index Documents** button in the sidebar to process the text, embed it, and store it in ChromaDB. Wait for the success message.
3. **Ask**: Enter a question in the main chat box and click **Submit**.
4. **Verify Sources**: Read the generated answer and click on the expandable **Sources** sections below it to see exactly which file and page the information came from!
