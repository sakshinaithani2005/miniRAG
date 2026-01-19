"""
Document processing module for Mini RAG app.
Handles document loading, chunking, and text extraction.
"""

import tempfile
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config


def load_document_from_file(uploaded_file) -> List[Document]:
    """
    Load document from uploaded file (PDF or TXT).
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        List of Document objects
    """
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load based on file type
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        
        documents = loader.load()
        return documents
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_text_from_string(text: str) -> List[Document]:
    """
    Create document from pasted text string.
    
    Args:
        text: Raw text content
    
    Returns:
        List with single Document object
    """
    return [Document(
        page_content=text,
        metadata={"source": "pasted_text"}
    )]


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks with overlap.
    
    Args:
        documents: List of Document objects
    
    Returns:
        List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=Config.CHUNK_SEPARATORS
    )
    
    return splitter.split_documents(documents)


def process_documents(
    uploaded_file=None,
    input_text: str = None
) -> List[Document]:
    """
    Complete document processing pipeline: load → chunk.
    
    Args:
        uploaded_file: Optional uploaded file
        input_text: Optional text string
    
    Returns:
        List of processed, chunked Document objects
    
    Raises:
        ValueError: If neither file nor text provided
    """
    if not uploaded_file and not input_text:
        raise ValueError("Either uploaded_file or input_text must be provided")
    
    # Load documents
    if uploaded_file:
        documents = load_document_from_file(uploaded_file)
    else:
        documents = load_text_from_string(input_text)
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    return chunks
