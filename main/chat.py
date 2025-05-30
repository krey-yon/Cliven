import requests
import json
import logging
from typing import List, Dict, Any, Optional
from utils.vectordb import ChromaDBManager
from utils.embedder import TextEmbedder
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Chat engine for RAG-based PDF conversation using ChromaDB and Ollama
    """

    def __init__(
        self,
        model_name: str = "tinyllama:chat",
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        chromadb_host: str = "localhost",
        chromadb_port: int = 8000,
        max_context_chunks: int = 5,
    ):
        """
        Initialize the chat engine

        Args:
            model_name (str): Ollama model name
            ollama_host (str): Ollama host (use 'ollama' in Docker)
            ollama_port (int): Ollama port
            chromadb_host (str): ChromaDB host (use 'chromadb' in Docker)
            chromadb_port (int): ChromaDB port
            max_context_chunks (int): Maximum number of context chunks to retrieve
        """
        self.model_name = model_name
        self.ollama_host = os.getenv("OLLAMA_HOST", ollama_host)
        self.ollama_port = int(os.getenv("OLLAMA_PORT", ollama_port))
        self.max_context_chunks = max_context_chunks

        # Initialize components
        self.db_manager = None
        self.embedder = None
        self.ollama_url = f"http://{self.ollama_host}:{self.ollama_port}"

        # Initialize ChromaDB and embedder
        self._initialize_components(chromadb_host, chromadb_port)

    def _initialize_components(self, chromadb_host: str, chromadb_port: int):
        """Initialize ChromaDB and embedder components"""
        try:
            # Initialize ChromaDB manager
            self.db_manager = ChromaDBManager(host=chromadb_host, port=chromadb_port)
            logger.info("ChromaDB manager initialized")

            # Initialize embedder for query embeddings
            self.embedder = TextEmbedder()
            logger.info("Text embedder initialized")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise Exception(f"Chat engine initialization failed: {e}")

    def _get_relevant_context(self, question: str) -> str:
        """
        Retrieve relevant context from ChromaDB based on question

        Args:
            question (str): User's question

        Returns:
            str: Relevant context from PDF documents
        """
        try:
            # Create embedding for the question
            question_embedding = self.embedder.create_embeddings([question])[0]

            # Search for similar chunks
            results = self.db_manager.search_similar(
                query_text=question,
                query_embedding=question_embedding,
                n_results=self.max_context_chunks,
            )

            # Combine relevant documents into context
            context_chunks = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, (doc, metadata, distance) in enumerate(
                zip(documents, metadatas, distances)
            ):
                source_file = metadata.get("source_file", "unknown")
                chunk_info = f"[Source: {source_file}, Relevance: {1-distance:.2f}]"
                context_chunks.append(f"{chunk_info}\n{doc}")

            context = "\n\n---\n\n".join(context_chunks)

            if not context.strip():
                return "No relevant context found in the uploaded documents."

            logger.info(f"Retrieved {len(context_chunks)} relevant context chunks")
            return context

        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return "Error retrieving context from documents."

    def _generate_response(self, context: str, question: str) -> str:
        """
        Generate response using Ollama

        Args:
            context (str): Relevant context from documents
            question (str): User's question

        Returns:
            str: Generated response
        """
        try:
            # Create the prompt
            prompt = self._create_prompt(context, question)

            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 4096},
            }

            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "No response generated.")
                logger.info("Successfully generated response")
                return answer.strip()
            else:
                logger.error(f"Ollama request failed: {response.status_code}")
                return "Sorry, I couldn't generate a response. Please try again."

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error with Ollama: {e}")
            return "Sorry, I'm having trouble connecting to the AI model. Please check if Ollama is running."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating the response."

    def _create_prompt(self, context: str, question: str) -> str:
        """
        Create a well-structured prompt for the LLM

        Args:
            context (str): Relevant context from documents
            question (str): User's question

        Returns:
            str: Formatted prompt
        """
        # Clean the context to remove metadata noise
        clean_context = context.replace("[Source:", "\nSource:").replace(
            "Relevance:", "Relevance:"
        )

        prompt = f"""Answer the question using only the information provided below. Give a direct, helpful response.

{clean_context}

Question: {question}

Answer the question directly without repeating the question or mentioning instructions:"""

        return prompt

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Main method to ask a question and get an answer

        Args:
            question (str): User's question

        Returns:
            Dict[str, Any]: Response with answer and metadata
        """
        try:
            if not question.strip():
                return {
                    "answer": "Please provide a valid question.",
                    "context_found": False,
                    "error": None,
                }

            logger.info(f"Processing question: {question[:100]}...")

            # Get relevant context
            context = self._get_relevant_context(question)

            # Generate response
            answer = self._generate_response(context, question)

            return {
                "answer": answer,
                "context_found": context
                != "No relevant context found in the uploaded documents.",
                "context_chunks": self.max_context_chunks,
                "model_used": self.model_name,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error in ask method: {e}")
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "context_found": False,
                "error": str(e),
            }

    def chat_session(self):
        """
        Interactive chat session
        """
        print("🤖 Cliven Chat - Ask questions about your PDF documents!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.\n")

        while True:
            try:
                question = input("You: ").strip()

                if question.lower() in ["quit", "exit", "bye", "q"]:
                    print("👋 Goodbye! Thanks for using Cliven!")
                    break

                if not question:
                    continue

                print("🤔 Thinking...")

                # Get response
                response = self.ask(question)

                # Display answer
                print(f"\n🤖 Cliven: {response['answer']}\n")

                # Show metadata if helpful
                if not response["context_found"]:
                    print(
                        "💡 Tip: Make sure you've ingested PDF documents first using 'cliven ingest <pdf_path>'\n"
                    )

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")

    def health_check(self) -> Dict[str, Any]:
        """
        Check if all components are working

        Returns:
            Dict[str, Any]: Health status of all components
        """
        health_status = {"overall_status": "healthy", "components": {}}

        # Check ChromaDB
        try:
            db_health = self.db_manager.health_check()
            health_status["components"]["chromadb"] = db_health
        except Exception as e:
            health_status["components"]["chromadb"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_status"] = "unhealthy"

        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]

                health_status["components"]["ollama"] = {
                    "status": "healthy",
                    "available_models": model_names,
                    "current_model": self.model_name,
                    "model_available": self.model_name in model_names,
                }

                if self.model_name not in model_names:
                    health_status["overall_status"] = "warning"
                    health_status["components"]["ollama"][
                        "warning"
                    ] = f"Model {self.model_name} not found"
            else:
                health_status["components"]["ollama"] = {
                    "status": "unhealthy",
                    "error": "Service unavailable",
                }
                health_status["overall_status"] = "unhealthy"

        except Exception as e:
            health_status["components"]["ollama"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_status"] = "unhealthy"

        return health_status


# Convenience function
def create_chat_engine(model_name: str = "tinyllama:chat") -> ChatEngine:
    """
    Create a chat engine instance

    Args:
        model_name (str): Ollama model name

    Returns:
        ChatEngine: Initialized chat engine
    """
    return ChatEngine(model_name=model_name)


# Example usage
if __name__ == "__main__":
    try:
        chat_engine = create_chat_engine()
        chat_engine.chat_session()
    except Exception as e:
        print(f"Failed to start chat engine: {e}")
