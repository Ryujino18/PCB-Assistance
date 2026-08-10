import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

CHROMA_DIR = "chroma_db"

def get_answer(query):
    # Setup embeddings (free local model)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(CHROMA_DIR):
        return "Vector database not found. Please upload and index documents first.", []

    # 1. Load persisted ChromaDB
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 2. Retrieve documents
    docs = retriever.invoke(query)
    
    if not docs:
        return "The information is not available in the uploaded documents.", []
        
    # Format context with source citations (file name and page)
    context_blocks = []
    for doc in docs:
        file_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
        page_num = doc.metadata.get('page', 0) + 1
        context_blocks.append(f"Source: {file_name} (Page {page_num})\nContent: {doc.page_content}")
    
    context = "\n\n".join(context_blocks)
    
    # 3. Use Answering Model (Groq Llama 3 instead of GPT-4o)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    
    # 4. Prompt with strict instructions for honest refusal
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the context does not contain the answer, reply exactly with 'The information is not available in the uploaded documents.' Do not guess or use outside knowledge. Keep the answer concise."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    
    # Generate answer
    response = chain.invoke({"context": context, "question": query})
    
    return response.content, docs
