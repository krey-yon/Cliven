import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    ChromaDB manager for storing and retrieving document embeddings
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        Initialize ChromaDB client

        Args:
            host (str): ChromaDB host (default: localhost, use 'chromadb' in Docker)
            port (int): ChromaDB port (default: 8000)
        """
        self.host = host
        self.port = port
        self.client = None
        self.collection = None
        self.collection_name = "pdf_documents"
        self._connect()

    def _connect(self):
        """Connect to ChromaDB"""
        try:
            # Use environment variables if available (for Docker)
            chroma_host = os.getenv("CHROMA_HOST", self.host)
            chroma_port = int(os.getenv("CHROMA_PORT", self.port))

            logger.info(f"Connecting to ChromaDB at {chroma_host}:{chroma_port}")

            # Connect to ChromaDB
            self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

            # Test connection
            self.client.heartbeat()
            logger.info("Successfully connected to ChromaDB")

            # Get or create collection
            self._get_or_create_collection()

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise Exception(f"ChromaDB connection failed: {e}")

    def _get_or_create_collection(self):
        """Get or create the documents collection"""
        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except:
                # Create new collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "PDF document chunks with embeddings"},
                )
                logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise Exception(f"Collection setup failed: {e}")

    def store_embeddings(self, embedding_data: Dict[str, Any]) -> bool:
        """
        Store embeddings in ChromaDB

        Args:
            embedding_data (Dict[str, Any]): Embedding data from embedder.py

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            # Validate embedding data
            required_keys = ["ids", "texts", "embeddings", "metadatas"]
            for key in required_keys:
                if key not in embedding_data:
                    raise Exception(f"Missing required key: {key}")

            ids = embedding_data["ids"]
            texts = embedding_data["texts"]
            embeddings = embedding_data["embeddings"]
            metadatas = embedding_data["metadatas"]

            # Validate data consistency
            if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
                raise Exception("Inconsistent data lengths")

            logger.info(f"Storing {len(ids)} embeddings in ChromaDB")

            # Store in ChromaDB
            self.collection.add(
                ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
            )

            logger.info(f"Successfully stored {len(ids)} embeddings")
            return True

        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return False

    def search_similar(
        self, query_text: str, query_embedding: List[float], n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Search for similar documents using embedding

        Args:
            query_text (str): Original query text (for logging)
            query_embedding (List[float]): Query embedding vector
            n_results (int): Number of results to return

        Returns:
            Dict[str, Any]: Search results
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            logger.info(f"Searching for similar documents: '{query_text[:50]}...'")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            logger.info(f"Found {len(results['documents'][0])} similar documents")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def search_similar_chunks(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the query

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of similar chunks with metadata
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            # Create embedding for the query (you'll need to import the embedder)
            from utils.embedder import TextEmbedder

            embedder = TextEmbedder()
            query_embedding = embedder.create_embeddings([query])[0]

            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=["documents", "metadatas", "distances"],
            )

            chunks = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    chunk = {
                        "text": doc,
                        "metadata": (
                            results["metadatas"][0][i]
                            if results["metadatas"] and results["metadatas"][0]
                            else {}
                        ),
                        "distance": (
                            results["distances"][0][i]
                            if results["distances"] and results["distances"][0]
                            else 0.0
                        ),
                    }
                    chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []

    def list_documents(self) -> Dict[str, Any]:
        """
        List all documents in the collection

        Returns:
            Dict[str, Any]: Collection information
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            # Get collection info
            count = self.collection.count()

            # Get sample documents
            sample_results = self.collection.get(limit=10, include=["metadatas"])

            # Extract unique source files
            source_files = set()
            for metadata in sample_results.get("metadatas", []):
                source_file = metadata.get("source_file", "unknown")
                source_files.add(source_file)

            return {
                "total_chunks": count,
                "unique_documents": len(source_files),
                "source_files": list(source_files),
                "collection_name": self.collection_name,
            }

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {"total_chunks": 0, "unique_documents": 0, "source_files": []}

    def delete_document(self, source_file: str) -> bool:
        """
        Delete all chunks from a specific document

        Args:
            source_file (str): Name of the source file to delete

        Returns:
            bool: True if successful
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            # Find all chunks from this document
            results = self.collection.get(
                where={"source_file": source_file}, include=["metadatas"]
            )

            if not results["ids"]:
                logger.warning(f"No chunks found for document: {source_file}")
                return False

            # Delete all chunks
            self.collection.delete(ids=results["ids"])

            logger.info(
                f"Deleted {len(results['ids'])} chunks from document: {source_file}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {source_file}: {e}")
            return False

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection

        Returns:
            bool: True if successful
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            # Get all IDs
            all_data = self.collection.get()

            if not all_data["ids"]:
                logger.info("Collection is already empty")
                return True

            # Delete all
            self.collection.delete(ids=all_data["ids"])

            logger.info(f"Cleared {len(all_data['ids'])} chunks from collection")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get detailed collection statistics

        Returns:
            Dict[str, Any]: Collection statistics
        """
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}

            count = self.collection.count()

            if count == 0:
                return {"total_chunks": 0, "documents": []}

            # Get all metadata to analyze
            all_data = self.collection.get(include=["metadatas"])

            # Analyze documents
            doc_stats = {}
            for metadata in all_data.get("metadatas", []):
                source_file = metadata.get("source_file", "unknown")
                if source_file not in doc_stats:
                    doc_stats[source_file] = {
                        "chunk_count": 0,
                        "total_size": 0,
                        "file_size": metadata.get("file_size", 0),
                    }
                doc_stats[source_file]["chunk_count"] += 1
                doc_stats[source_file]["total_size"] += metadata.get("chunk_size", 0)

            return {
                "total_chunks": count,
                "total_documents": len(doc_stats),
                "documents": doc_stats,
                "collection_name": self.collection_name,
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """
        Check ChromaDB connection health

        Returns:
            Dict[str, Any]: Health status
        """
        try:
            # Test client connection
            self.client.heartbeat()

            # Test collection
            count = self.collection.count() if self.collection else 0

            return {
                "status": "healthy",
                "host": self.host,
                "port": self.port,
                "collection": self.collection_name,
                "document_count": count,
                "timestamp": time.time(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "host": self.host,
                "port": self.port,
                "timestamp": time.time(),
            }


# Convenience functions for easy integration
def store_embeddings_to_chromadb(
    embedding_data: Dict[str, Any], host: str = None, port: int = 8000
) -> bool:
    """
    Convenience function to store embeddings

    Args:
        embedding_data (Dict[str, Any]): Embedding data from embedder.py
        host (str): ChromaDB host (auto-detect if None)
        port (int): ChromaDB port

    Returns:
        bool: True if successful
    """
    try:
        # Auto-detect host if not specified
        if host is None:
            # Check if running in Docker (look for .dockerenv file)
            if os.path.exists("/.dockerenv"):
                host = "chromadb"  # Docker container name
            else:
                host = "localhost"  # Local development

        db_manager = ChromaDBManager(host=host, port=port)
        return db_manager.store_embeddings(embedding_data)
    except Exception as e:
        logger.error(f"Failed to store embeddings: {e}")
        return False


def get_chromadb_manager(host: str = "chromadb", port: int = 8000) -> ChromaDBManager:
    """
    Get ChromaDB manager instance

    Args:
        host (str): ChromaDB host
        port (int): ChromaDB port

    Returns:
        ChromaDBManager: Database manager instance
    """
    return ChromaDBManager(host=host, port=port)


# Example usage
if __name__ == "__main__":
    # Test the ChromaDB connection
    try:
        db_manager = ChromaDBManager(host="localhost", port=8000)
        health = db_manager.health_check()
        print(f"ChromaDB Health: {health}")

        stats = db_manager.get_collection_stats()
        print(f"Collection Stats: {stats}")

    except Exception as e:
        print(f"Test failed: {e}")
