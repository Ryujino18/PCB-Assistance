import os
import glob
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
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(docs)
    
    # 3. Embeddings (Using a free local HuggingFace model instead of OpenAI)
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
