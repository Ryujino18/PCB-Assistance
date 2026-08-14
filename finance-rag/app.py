import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.utils.pydantic")
import streamlit as st
from ingest import ingest_pdfs
from rag import get_answer

DATA_DIR = "data"

# Setup Streamlit Interface
st.set_page_config(page_title="Finance RAG System", layout="wide")
st.title("Finance RAG System (OpenAI Edition)")

os.makedirs(DATA_DIR, exist_ok=True)

# 1. Upload & Index - Sidebar
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("1. Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Save Uploaded Files"):
        if uploaded_files:
            for file in uploaded_files:
                file_path = os.path.join(DATA_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s) to {DATA_DIR}/.")
        else:
            st.warning("Please upload files first.")
            
    st.markdown("---")
    
    if st.button("2. Index Documents"):
        with st.spinner("Processing documents (chunking & embedding)..."):
            num_files, num_chunks = ingest_pdfs()
            if num_files > 0:
                st.success(f"Success! {num_files} files processed, {num_chunks} chunks stored in ChromaDB.")
            else:
                st.error("No files found in data directory. Upload and save first.")

# 3. Ask - Main Content
st.header("Ask a Question")
query = st.text_input("Enter your question here:")

if st.button("Submit"):
    if query:
        with st.spinner("Finding answer..."):
            # 4. Answer
            answer, sources = get_answer(query)
            
            st.subheader("Answer:")
            st.write(answer)
            
            # 5. Sources
            if sources and "The information is not available" not in answer:
                st.markdown("---")
                st.subheader("Sources used:")
                for i, doc in enumerate(sources):
                    file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page_num = doc.metadata.get('page', 0) + 1
                    with st.expander(f"Source {i+1}: {file_name} (Page {page_num})"):
                        st.write(doc.page_content)
    else:
        st.warning("Please enter a question.")
