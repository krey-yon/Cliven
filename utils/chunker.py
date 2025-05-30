# chunker.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from pathlib import Path
import logging

# 🔁 CHANGE THIS: Import extract_text_from_pdf from your parser.py file
# Update the import based on your project structure
from parser import extract_text_from_pdf  # <-- CHANGE THIS PATH IF NEEDED

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Chunk text using LangChain's RecursiveCharacterTextSplitter
    
    Args:
        text (str): Text to be chunked
        chunk_size (int): Maximum size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        List[Dict[str, Any]]: List of text chunks with metadata
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_text(text)

        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i + 1,
                "text": chunk,
                "chunk_size": len(chunk),
                "metadata": {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_length": len(text)
                }
            }
            chunked_documents.append(chunk_data)

        logger.info(f"Successfully created {len(chunked_documents)} chunks")
        return chunked_documents

    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise Exception(f"Failed to chunk text: {e}")

def parse_pdf_with_chunking(pdf_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Complete PDF processing pipeline: extract text and chunk it
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Maximum size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        List[Dict[str, Any]]: List of text chunks with metadata
    """
    try:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise Exception(f"PDF file not found: {pdf_path}")

        if not pdf_file.suffix.lower() == '.pdf':
            raise Exception(f"File must be a PDF: {pdf_path}")

        logger.info(f"Starting PDF processing: {pdf_file.name}")

        extracted_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(extracted_text, chunk_size, overlap)

        for chunk in chunks:
            chunk["metadata"].update({
                "source_file": pdf_file.name,
                "source_path": str(pdf_file),
                "file_size": pdf_file.stat().st_size
            })

        logger.info(f"PDF processing complete: {len(chunks)} chunks created")
        return chunks

    except Exception as e:
        logger.error(f"Error in PDF processing pipeline: {e}")
        raise Exception(f"PDF processing failed: {e}")

def preview_chunks(chunks: List[Dict[str, Any]], max_preview: int = 3) -> None:
    """
    Preview the first few chunks for debugging
    
    Args:
        chunks (List[Dict[str, Any]]): List of text chunks
        max_preview (int): Maximum number of chunks to preview
    """
    print(f"\n📋 Preview of {min(len(chunks), max_preview)} chunks:")
    print("=" * 50)

    for i, chunk in enumerate(chunks[:max_preview]):
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"Size: {chunk['chunk_size']} characters")
        print(f"Preview: {chunk['text'][:100]}...")
        print("-" * 30)

    if len(chunks) > max_preview:
        print(f"\n... and {len(chunks) - max_preview} more chunks")
