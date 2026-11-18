"""
Core RAG (Retrieval-Augmented Generation) Pipeline Module.

This module handles:
1. Document text extraction (PDF & TXT)
2. Text chunking with overlap
3. Vector embedding generation with Sentence Transformers
4. In-memory Vector indexing with FAISS
5. Context retrieval and grounded QA generation using Google Gemini LLM
"""

import os
from io import BytesIO
from pypdf import PdfReader

# LangChain components
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def extract_text_from_file(uploaded_file) -> str:
    """
    Extracts raw text from an uploaded file (PDF or TXT).
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
    
    Returns:
        Extracted text as a string.
    """
    filename = uploaded_file.name.lower()
    text = ""

    if filename.endswith(".pdf"):
        # Read PDF using pypdf
        reader = PdfReader(BytesIO(uploaded_file.read()))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif filename.endswith(".txt"):
        # Read text file with utf-8 decoding (fallback to latin-1)
        raw_bytes = uploaded_file.read()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1")
    else:
        raise ValueError(f"Unsupported file format: {filename}. Please upload a .pdf or .txt file.")

    return text.strip()


def split_text_into_chunks(text: str, source_name: str = "document", chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Splits text into smaller, overlapping chunks suitable for embedding.
    
    Args:
        text: Raw text to split.
        source_name: Name of the source file for metadata.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.
        
    Returns:
        List of LangChain Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    # Wrap in LangChain Document objects with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_id": idx}
        )
        for idx, chunk in enumerate(chunks)
    ]
    return documents


def get_embedding_model():
    """
    Loads the HuggingFace Sentence Transformers embedding model.
    Model: 'sentence-transformers/all-MiniLM-L6-v2' (fast, lightweight, 384 dimensions).
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vector_store(documents, embedding_model):
    """
    Embeds documents and indexes them in an in-memory FAISS vector store.
    
    Args:
        documents: List of LangChain Document objects.
        embedding_model: Embedding model instance.
        
    Returns:
        FAISS vector store instance.
    """
    if not documents:
        raise ValueError("Cannot create vector store from empty document list.")
    
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store


def answer_question(vector_store, question: str, api_key: str, k: int = 3):
    """
    Performs retrieval and generation for a user query.
    
    1. Finds top-k most relevant chunks using similarity search.
    2. Builds an augmented prompt containing context and strict anti-hallucination rules.
    3. Calls Google Gemini LLM to generate the answer.
    
    Args:
        vector_store: FAISS vector database.
        question: User query string.
        api_key: Google Gemini API key.
        k: Number of relevant chunks to retrieve.
        
    Returns:
        dict with keys: 'answer', 'source_documents'
    """
    # 1. Similarity search in FAISS
    relevant_docs = vector_store.similarity_search(question, k=k)

    if not relevant_docs:
        return {
            "answer": "No relevant context found in the uploaded documents to answer your question.",
            "source_documents": []
        }

    # 2. Combine retrieved chunks into context text
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

    # 3. Define prompt template with strict instructions
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful and strict AI document answering assistant.
Answer the user's question ONLY using the factual context provided below.
If the answer is NOT present or cannot be directly deduced from the context, respond exactly with:
"I could not find the answer to this question in the uploaded document(s)."
Do NOT speculate, assume, or make up information outside of the context.

Retrieved Context:
{context}

User Question:
{question}

Answer:"""
    )

    # 4. Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1,  # Low temperature to minimize hallucination
        max_output_tokens=1024
    )

    # 5. Format prompt and invoke LLM
    formatted_prompt = prompt_template.format(context=context_text, question=question)
    response = llm.invoke(formatted_prompt)
    
    answer_text = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer_text,
        "source_documents": relevant_docs
    }
