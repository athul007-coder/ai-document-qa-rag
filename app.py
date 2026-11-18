"""
Streamlit Web Application for AI Document Question Answering using LLM & RAG.

Author: AI Assistant
Framework: Streamlit, LangChain, FAISS, Sentence Transformers, Google Gemini
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load local .env file if available
load_dotenv()

from rag_pipeline import (
    extract_text_from_file,
    split_text_into_chunks,
    get_embedding_model,
    create_vector_store,
    answer_question,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Document QA System (RAG)",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .chunk-box {
        background-color: #F8FAFC;
        border-left: 4px solid #6366F1;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 12px;
        background-color: #EEF2FF;
        color: #4F46E5;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialize Session States
# -----------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Cache embedding model to avoid re-downloading across re-renders
@st.cache_resource(show_spinner="Loading Sentence Transformer embedding model...")
def load_embeddings():
    return get_embedding_model()

embedding_model = load_embeddings()

# -----------------------------------------------------------------------------
# Sidebar: Setup & Document Ingestion
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # 1. API Key Input
    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        help="Get a free API key at https://aistudio.google.com/app/apikey"
    )
    api_key = api_key_input.strip()

    st.markdown("---")
    st.subheader("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload one or more documents to query."
    )

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", min_value=200, max_value=2000, value=800, step=100)
    with col2:
        chunk_overlap = st.number_input("Overlap", min_value=0, max_value=500, value=150, step=50)

    process_btn = st.button("🚀 Process Documents", type="primary", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.error("Please upload at least one PDF or TXT file.")
        else:
            with st.spinner("Processing & building FAISS vector index..."):
                all_documents = []
                file_names = []
                
                try:
                    for file in uploaded_files:
                        text = extract_text_from_file(file)
                        if text:
                            chunks = split_text_into_chunks(
                                text=text,
                                source_name=file.name,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            all_documents.extend(chunks)
                            file_names.append(file.name)
                        else:
                            st.warning(f"File {file.name} was empty or text could not be extracted.")

                    if all_documents:
                        st.session_state.vector_store = create_vector_store(all_documents, embedding_model)
                        st.session_state.processed_files = file_names
                        st.session_state.total_chunks = len(all_documents)
                        st.session_state.chat_history = []  # Reset chat on new document
                        st.success(f"Successfully indexed {len(file_names)} file(s) into {len(all_documents)} chunks!")
                    else:
                        st.error("No valid text found in uploaded files.")
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")

    # Display Status & Metrics
    if st.session_state.vector_store is not None:
        st.markdown("---")
        st.subheader("📊 Index Status")
        st.markdown(f"**Indexed Files:** `{', '.join(st.session_state.processed_files)}`")
        st.markdown(f"**Total Chunks:** `{st.session_state.total_chunks}`")
        st.markdown(f"**Embedding Model:** `all-MiniLM-L6-v2`")
        st.markdown(f"**Vector Store:** `FAISS (In-Memory)`")

        if st.button("🧹 Clear & Reset Index", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.processed_files = []
            st.session_state.total_chunks = 0
            st.session_state.chat_history = []
            st.rerun()

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">📚 AI Document QA System (RAG)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about your uploaded documents powered by LangChain, FAISS, and Gemini LLM.</div>',
    unsafe_allow_html=True
)

# Case 1: No documents processed yet
if st.session_state.vector_store is None:
    st.info("👈 **Get Started**: Upload a PDF or TXT file using the sidebar and click **'Process Documents'**.")
    
    with st.expander("💡 How this RAG system works (Beginner Explanation)", expanded=True):
        st.markdown("""
        **RAG (Retrieval-Augmented Generation)** prevents AI hallucinations by grounding responses in your documents:
        
        1. **Extraction**: Reads text from your uploaded PDF or TXT documents.
        2. **Chunking**: Splits large texts into smaller paragraphs (e.g., 800 characters) so each piece fits nicely in memory.
        3. **Embeddings**: An AI embedding model (`Sentence Transformers`) converts each chunk into numbers representing semantic meaning.
        4. **Vector Store**: Embeddings are indexed in an in-memory database (**FAISS**) for ultra-fast similarity search.
        5. **Retrieval**: When you ask a question, FAISS finds the top-3 most relevant paragraphs.
        6. **Answer Generation**: The **Gemini LLM** receives only the retrieved paragraphs and the question, answering strictly from the text!
        """)

# Case 2: Documents are indexed and ready for Question-Answering
else:
    # Quick Document Badge
    files_str = ", ".join(st.session_state.processed_files)
    st.markdown(f"<span class='badge-pill'>Active Documents: {files_str}</span> <span class='badge-pill'>Chunks: {st.session_state.total_chunks}</span>", unsafe_allow_html=True)

    # Question Input
    user_question = st.text_input(
        "Enter your question based on the document:",
        placeholder="e.g., What is the main conclusion of the document?",
        key="question_input"
    )

    k_chunks = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=6, value=3)

    if st.button("🔍 Get Answer", type="primary") or (user_question and st.session_state.get("trigger_ask")):
        if not user_question.strip():
            st.warning("Please type a question before submitting.")
        elif not api_key:
            st.error("Google Gemini API Key is required. Please enter it in the sidebar.")
        else:
            with st.spinner("Retrieving relevant context and querying Gemini LLM..."):
                try:
                    result = answer_question(
                        vector_store=st.session_state.vector_store,
                        question=user_question.strip(),
                        api_key=api_key,
                        k=k_chunks
                    )

                    # Save to chat history
                    st.session_state.chat_history.insert(0, {
                        "question": user_question.strip(),
                        "answer": result["answer"],
                        "sources": result["source_documents"]
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

    # Display Results / Chat History
    if st.session_state.chat_history:
        st.markdown("### 💬 Answers & Explanations")
        
        for idx, item in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**❓ Question:** {item['question']}")
                st.markdown(f"**🤖 Answer:**\n{item['answer']}")
                
                # Expandable Source Citations
                with st.expander(f"🔎 View {len(item['sources'])} Retrieved Context Chunks (Sources)", expanded=(idx == 0)):
                    for s_idx, doc in enumerate(item["sources"]):
                        source_file = doc.metadata.get("source", "Unknown")
                        chunk_id = doc.metadata.get("chunk_id", s_idx)
                        st.markdown(f"""
                        <div class="chunk-box">
                            <strong>Source:</strong> <code>{source_file}</code> | <strong>Chunk ID:</strong> <code>{chunk_id}</code><br/>
                            <p style="margin-top: 6px; white-space: pre-wrap;">{doc.page_content}</p>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("---")
