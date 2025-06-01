import requests
import json
from typing import List, Dict, Any, Optional
import subprocess
import os


def get_available_ollama_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        result = subprocess.run(
            ["docker", "exec", "cliven_ollama", "ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]  # First column is model name
                    models.append(model_name)
            return models
        return []
    except Exception:
        return []


def select_best_available_ollama_model() -> str:
    """Select the best available model based on priority"""
    available_models = get_available_ollama_models()

    # Priority order: gemma3:4b first (better performance), then gemma2:2b
    priority_models = ["gemma3:4b", "gemma2:2b"]

    for model in priority_models:
        if model in available_models:
            return model

    # If none of the priority models are available, return first available or default
    if available_models:
        return available_models[0]

    return "gemma2:2b"  # Default fallback


class ChatEngine:
    """
    Chat engine for RAG-based PDF conversation using ChromaDB and Ollama
    """

    def __init__(
        self,
        model_name: Optional[str] = None,  # Changed to Optional
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        chromadb_host: str = "localhost",
        chromadb_port: int = 8000,
        max_context_chunks: int = 5,
    ):
        """
        Initialize the chat engine

        Args:
            model_name (Optional[str]): Ollama model name (auto-selected if None)
            ollama_host (str): Ollama host (use 'ollama' in Docker)
            ollama_port (int): Ollama port
            chromadb_host (str): ChromaDB host (use 'chromadb' in Docker)
            chromadb_port (int): ChromaDB port
            max_context_chunks (int): Maximum number of context chunks to retrieve
        """
        # Auto-select model if not specified
        if model_name is None:
            self.model_name = select_best_available_ollama_model()
        else:
            # Check if specified model is available
            available_models = get_available_ollama_models()
            if model_name in available_models:
                self.model_name = model_name
            else:
                self.model_name = select_best_available_ollama_model()

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
            from utils.vectordb import ChromaDBManager
            from utils.embedder import TextEmbedder

            self.db_manager = ChromaDBManager(host=chromadb_host, port=chromadb_port)

            # Initialize embedder for query embeddings
            self.embedder = TextEmbedder()

        except Exception as e:
            raise Exception(f"Chat engine initialization failed: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        available_models = get_available_ollama_models()
        return {
            "current_model": self.model_name,
            "available_models": available_models,
            "model_available": self.model_name in available_models,
            "model_priority": "high" if self.model_name == "gemma3:4b" else "standard",
        }

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

            return context

        except Exception as e:
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

            # Adjust parameters based on model
            if self.model_name == "gemma3:4b":
                # Higher performance settings for mgemma3:4b
                options = {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 8192,  # Higher context window
                    "repeat_penalty": 1.1,
                }
                timeout = 120  # Longer timeout for larger model
            else:
                # Standard settings for smaller models
                options = {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "repeat_penalty": 1.1,
                }
                timeout = 60

            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }

            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "No response generated.")
                return answer.strip()
            else:
                return "Sorry, I couldn't generate a response. Please try again."

        except requests.exceptions.RequestException as e:
            return "Sorry, I'm having trouble connecting to the AI model. Please check if Ollama is running."
        except Exception as e:
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

        # Check if question is a greeting
        greeting_keywords = [
            "hi",
            "hello",
            "hey",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        is_greeting = any(
            keyword in question.lower().strip() for keyword in greeting_keywords
        )

        # Enhanced prompt for better models (gemma3:4b)
        if self.model_name == "gemma3:4b":
            prompt = f"""You are Cliven, an intelligent PDF assistant. Your job is to help users understand the content of PDF documents by answering their questions using the provided context. Analyze the question carefully and respond appropriately.

Context from documents:
{clean_context}

Question: {question}

Instructions:
1. If the question is a **greeting** (e.g., "hi", "hello", "hey"), respond with a warm, friendly greeting as Cliven and briefly mention your capabilities.
2. If the question is **general-purpose or unrelated to the document content**, acknowledge that it's outside the PDF context. Preface your response with: "⚠️ This response is based on general knowledge, not the document content."
3. If the question is about **something in the document context**, provide a clear, detailed, and well-structured answer using ONLY the provided context. Include relevant details and organize your response logically.
4. If the context is **insufficient or doesn't contain the answer**, acknowledge this limitation by saying: "📄 The document does not provide enough information to fully answer this question."
5. Always be helpful, accurate, and concise in your responses.

Generate your response now:"""

        # Simplified prompt for lightweight models (gemma2:2b)
        else:
            prompt = f"""You are Cliven, a helpful assistant for answering questions about PDF documents. Use the provided context to answer questions accurately.

Context from documents:
{clean_context}

Question: {question}

Instructions:
- If greeting (like "hi", "hello"), greet warmly and mention you help with PDF questions
- If general knowledge question unrelated to the document, say: "⚠️ This response is based on general knowledge, not the document content."
- If document doesn't contain the answer, say: "📄 The document does not provide enough information to answer this question."
- Otherwise, answer clearly using only the document content
- Be helpful and concise

Answer:"""

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
                "model_info": self.get_model_info(),
                "error": None,
            }

        except Exception as e:
            return {
                "answer": "Sorry, I encountered an error while processing your question.",
                "context_found": False,
                "error": str(e),
            }

    def chat_session(self):
        """
        Interactive chat session
        """
        model_info = self.get_model_info()
        performance_indicator = "🚀" if model_info["model_priority"] == "high" else "⚡"

        print("🤖 Cliven Chat - Ask questions about your PDF documents!")
        print(f"{performance_indicator} Using model: {self.model_name}")
        if model_info["model_priority"] == "high":
            print("   🎯 High-performance mode for better responses")
        else:
            print("   ⚡ Fast response mode")
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
def create_chat_engine(model_name: Optional[str] = None) -> ChatEngine:
    """
    Create a chat engine instance with intelligent model selection

    Args:
        model_name (Optional[str]): Specific model name (auto-selected if None)

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
