from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    Text embedder using BAAI/bge-small-en-v1.5 model for creating embeddings
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedder with the specified model
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise Exception(f"Could not load embedding model {self.model_name}: {e}")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            if not self.model:
                raise Exception("Model not loaded")
            
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True if len(texts) > 10 else False
            )
            
            # Convert to list of lists for ChromaDB compatibility
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Successfully created {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise Exception(f"Failed to create embeddings: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors
        
        Returns:
            int: Embedding dimension
        """
        if not self.model:
            raise Exception("Model not loaded")
        return self.model.get_sentence_embedding_dimension()

def process_chunks_for_embedding(chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Prepare chunks for embedding by extracting texts and metadata
    
    Args:
        chunks (List[Dict[str, Any]]): List of text chunks from chunker.py
        
    Returns:
        Tuple[List[str], List[Dict[str, Any]], List[str]]: 
            - texts: List of chunk texts for embedding
            - metadatas: List of metadata dictionaries
            - ids: List of unique IDs for each chunk
    """
    try:
        texts = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Extract text content
            chunk_text = chunk.get("text", "")
            if not chunk_text.strip():
                logger.warning(f"Empty text in chunk {chunk.get('chunk_id', 'unknown')}")
                continue
            
            texts.append(chunk_text)
            
            # Prepare metadata for ChromaDB (remove nested dicts)
            metadata = {
                "chunk_id": chunk.get("chunk_id", 0),
                "chunk_size": chunk.get("chunk_size", 0),
                "chunk_index": chunk.get("metadata", {}).get("chunk_index", 0),
                "total_chunks": chunk.get("metadata", {}).get("total_chunks", 0),
                "source_file": chunk.get("metadata", {}).get("source_file", "unknown"),
                "source_path": chunk.get("metadata", {}).get("source_path", ""),
                "file_size": chunk.get("metadata", {}).get("file_size", 0)
            }
            metadatas.append(metadata)
            
            # Create unique ID for each chunk
            source_file = metadata.get("source_file", "unknown")
            chunk_id = chunk.get("chunk_id", 0)
            unique_id = f"{Path(source_file).stem}_chunk_{chunk_id}"
            ids.append(unique_id)
        
        logger.info(f"Prepared {len(texts)} chunks for embedding")
        return texts, metadatas, ids
        
    except Exception as e:
        logger.error(f"Error processing chunks for embedding: {e}")
        raise Exception(f"Failed to process chunks: {e}")

def create_embeddings_for_chunks(chunks: List[Dict[str, Any]], model_name: str = "BAAI/bge-small-en-v1.5") -> Dict[str, Any]:
    """
    Complete pipeline to create embeddings from chunks
    
    Args:
        chunks (List[Dict[str, Any]]): List of text chunks from chunker.py
        model_name (str): Name of the embedding model to use
        
    Returns:
        Dict[str, Any]: Dictionary containing texts, embeddings, metadatas, and ids ready for ChromaDB
    """
    try:
        if not chunks:
            raise Exception("No chunks provided")
        
        logger.info(f"Starting embedding creation for {len(chunks)} chunks")
        
        # Initialize embedder
        embedder = TextEmbedder(model_name)
        
        # Process chunks to extract texts, metadata, and IDs
        texts, metadatas, ids = process_chunks_for_embedding(chunks)
        
        if not texts:
            raise Exception("No valid texts found in chunks")
        
        # Create embeddings
        embeddings = embedder.create_embeddings(texts)
        
        # Prepare data for ChromaDB
        embedding_data = {
            "ids": ids,
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "embedding_dimension": embedder.get_embedding_dimension(),
            "model_name": model_name,
            "total_chunks": len(texts)
        }
        
        logger.info(f"Successfully created embeddings for {len(texts)} chunks")
        logger.info(f"Embedding dimension: {embedding_data['embedding_dimension']}")
        
        return embedding_data
        
    except Exception as e:
        logger.error(f"Error in embedding pipeline: {e}")
        raise Exception(f"Embedding creation failed: {e}")

def preview_embeddings(embedding_data: Dict[str, Any], max_preview: int = 3) -> None:
    """
    Preview embedding data for debugging
    
    Args:
        embedding_data (Dict[str, Any]): Embedding data dictionary
        max_preview (int): Maximum number of items to preview
    """
    print(f"\n🔍 Embedding Preview:")
    print("=" * 50)
    print(f"Total chunks: {embedding_data['total_chunks']}")
    print(f"Embedding dimension: {embedding_data['embedding_dimension']}")
    print(f"Model used: {embedding_data['model_name']}")
    print("\nSample chunks:")
    
    for i in range(min(len(embedding_data['texts']), max_preview)):
        print(f"\nChunk {i + 1}:")
        print(f"ID: {embedding_data['ids'][i]}")
        print(f"Text preview: {embedding_data['texts'][i][:100]}...")
        print(f"Embedding preview: {embedding_data['embeddings'][i][:5]}... (showing first 5 dims)")
        print(f"Metadata: {embedding_data['metadatas'][i]}")
        print("-" * 30)
    
    if len(embedding_data['texts']) > max_preview:
        print(f"\n... and {len(embedding_data['texts']) - max_preview} more chunks")

# Example usage function
def example_usage():
    """
    Example of how to use the embedder with chunked data
    """
    # This would typically be imported from chunker.py
    from chunker import parse_pdf_with_chunking
    
    # Example: Process a PDF and create embeddings
    pdf_path = "./sample.pdf"
    
    try:
        # Get chunks from chunker
        chunks = parse_pdf_with_chunking(pdf_path, chunk_size=1000, overlap=200)
        
        # Create embeddings
        embedding_data = create_embeddings_for_chunks(chunks)
        
        # Preview results
        preview_embeddings(embedding_data)
        
        return embedding_data
        
    except Exception as e:
        logger.error(f"Example usage failed: {e}")
        return None

if __name__ == "__main__":
    # Test the embedder
    example_usage()