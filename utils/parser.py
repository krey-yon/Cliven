# parser.py

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using pdfplumber

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text from the PDF

    Raises:
        Exception: If PDF cannot be read or processed
    """
    try:
        text_content = ""

        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"Processing PDF with {len(pdf.pages)} pages")

            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n\n--- Page {page_num} ---\n\n"
                        text_content += page_text
                        logger.debug(f"Extracted text from page {page_num}")
                    else:
                        logger.warning(f"No text found on page {page_num}")

                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {e}")
                    continue

        if not text_content.strip():
            raise Exception("No text content extracted from PDF")

        logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
        return text_content.strip()

    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise Exception(f"Failed to extract text from PDF: {e}")


def chunk_text(
    text: str, chunk_size: int = 1000, overlap: int = 200
) -> List[Dict[str, Any]]:
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
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # Priority order for splitting
        )

        # Split the text
        chunks = text_splitter.split_text(text)

        # Create chunks with metadata
        chunked_documents = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i + 1,
                "text": chunk,
                "chunk_size": len(chunk),
                "metadata": {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_length": len(text),
                },
            }
            chunked_documents.append(chunk_data)

        logger.info(f"Successfully created {len(chunked_documents)} chunks")
        return chunked_documents

    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise Exception(f"Failed to chunk text: {e}")


def parse_pdf_with_chunking(
    pdf_path: str, chunk_size: int = 1000, overlap: int = 200
) -> List[Dict[str, Any]]:
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
        # Validate PDF path
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise Exception(f"PDF file not found: {pdf_path}")

        if not pdf_file.suffix.lower() == ".pdf":
            raise Exception(f"File must be a PDF: {pdf_path}")

        logger.info(f"Starting PDF processing: {pdf_file.name}")

        # Step 1: Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        # Step 2: Chunk the extracted text
        chunks = chunk_text(extracted_text, chunk_size, overlap)

        # Add PDF metadata to each chunk
        for chunk in chunks:
            chunk["metadata"].update(
                {
                    "source_file": pdf_file.name,
                    "source_path": str(pdf_file),
                    "file_size": pdf_file.stat().st_size,
                }
            )

        logger.info(f"PDF processing complete: {len(chunks)} chunks created")
        return chunks

    except Exception as e:
        logger.error(f"Error in PDF processing pipeline: {e}")
        raise Exception(f"PDF processing failed: {e}")


# Utility function for testing
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


# Test function
if __name__ == "__main__":
    # Test the parser
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        try:
            chunks = parse_pdf_with_chunking(pdf_path)
            preview_chunks(chunks)
            print(f"\n✅ Successfully processed {len(chunks)} chunks")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Usage: python parser.py <pdf_path>")
