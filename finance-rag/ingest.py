import os
import glob
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

def ingest_pdfs():
    # Find all PDFs in the data directory
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        return 0, 0
    
    docs = []
    for file_path in pdf_files:
        # 1. Load PDFs
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
        
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    
    # Pre-process chunks to include source label for better retrieval matching (Stage 6)
    for chunk in chunks:
        file_name = os.path.basename(chunk.metadata.get('source', 'Unknown'))
        # Using filename to infer quarter if named like Infosys_Q1_FY26.pdf
        chunk.page_content = f"Source: {file_name}\n\n{chunk.page_content}"
    
    # Prevent duplication by clearing the old collection using ChromaDB's API (Stage 5)
    # This avoids file-lock issues caused by shutil.rmtree while ChromaDB is in use
    if os.path.exists(CHROMA_DIR):
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            client.delete_collection("langchain")
        except Exception:
            pass  # Collection may not exist yet, which is fine
    
    # 3. Embeddings (HuggingFace all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Vector Database (ChromaDB)
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_DIR
    )
    
    return len(pdf_files), len(chunks)

if __name__ == "__main__":
    ingest_pdfs()
