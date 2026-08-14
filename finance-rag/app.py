import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.utils.pydantic")
import streamlit as st
import requests

API_URL = "http://localhost:8000"
DATA_DIR = "data"

# Setup Streamlit Interface
st.set_page_config(page_title="Finance RAG System", layout="wide")
st.title("Finance RAG System (Groq Edition)")

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to check if backend is reachable and indexed
def check_backend_status():
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        return None
    return None

# 1. Upload & Index - Sidebar
with st.sidebar:
    st.header("Document Management")
    
    # Check backend status
    backend_status = check_backend_status()
    if backend_status is None:
        st.error("Backend API is not running. Please start it with `uvicorn api:app --reload`")
    
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
        if not backend_status:
            st.error("Backend is unavailable.")
        else:
            with st.spinner("Processing documents (chunking & embedding)..."):
                try:
                    # In this setup, we just trigger the backend's index.
                    # Our API is designed to accept files directly, but since we already saved them, 
                    # we can upload them from the DATA_DIR to the API endpoint.
                    files_to_upload = []
                    for filename in os.listdir(DATA_DIR):
                        if filename.endswith(".pdf"):
                            file_path = os.path.join(DATA_DIR, filename)
                            files_to_upload.append(
                                ("files", (filename, open(file_path, "rb"), "application/pdf"))
                            )
                    
                    if not files_to_upload:
                        st.error("No PDFs found to index. Upload and save first.")
                    else:
                        res = requests.post(f"{API_URL}/index", files=files_to_upload)
                        if res.status_code == 200:
                            data = res.json()
                            num_files = data.get("files_processed", 0)
                            num_chunks = data.get("chunks_stored", 0)
                            st.success(f"Success! {num_files} files processed, {num_chunks} chunks stored in ChromaDB.")
                            
                            # Close files
                            for _, (name, f, mime) in files_to_upload:
                                f.close()
                        else:
                            st.error(f"Error during indexing: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")

# 3. Ask - Main Content
st.header("Ask a Question")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}: {source['file_name']} (Page {source['page']})**")
                    st.text(source['content'])

# Chat input
if query := st.chat_input("Enter your question here:"):
    if not backend_status or backend_status.get("chunk_count", 0) == 0:
        st.warning("Please upload and index documents before asking questions.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.spinner("Finding answer..."):
            try:
                # 4. Answer via backend
                res = requests.post(f"{API_URL}/ask", json={"question": query, "top_k": 10})
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer received.")
                    sources = data.get("sources", [])
                    
                    # 5. Display Answer
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        
                        # 6. Display Sources
                        if sources and "The information is not available" not in answer:
                            with st.expander("View Sources"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}: {source['file_name']} (Page {source['page']})**")
                                    st.text(source['content'])
                                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources if "The information is not available" not in answer else []
                    })
                else:
                    st.error(f"Backend error: {res.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}")
