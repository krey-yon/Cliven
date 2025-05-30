import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_welcome():
    """Display welcome message and available commands"""
    print(
        """
🤖 Cliven - Chat with PDF CLI Tool
==================================

Available Commands:
  cliven ingest <pdf_path>           - Process and store PDF in vector database
  cliven chat                        - Start interactive chat session with existing docs
  cliven chat --repl <pdf_path>      - Process PDF and start interactive chat
  cliven list                        - List all processed documents
  cliven delete <doc_id>             - Delete a processed document
  cliven clear                       - Clear all processed documents
  cliven status                      - Show system status (ChromaDB, Ollama)
  cliven --help                      - Show detailed help
  cliven --version                   - Show version information

Examples:
  cliven ingest ./documents/manual.pdf
  cliven chat
  cliven chat --repl ./documents/manual.pdf
  cliven list
  
📘 Cliven is ready! Use any command above to get started.
    """
    )


def set_quiet_mode(enabled: bool = True):
    """Enable/disable quiet mode for chat (hide verbose logs)"""
    if enabled:
        # Suppress verbose logs from dependencies
        logging.getLogger("chat").setLevel(logging.ERROR)
        logging.getLogger("utils.embedder").setLevel(logging.ERROR)
        logging.getLogger("utils.vectordb").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("chromadb").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    else:
        # Restore normal logging
        logging.getLogger("chat").setLevel(logging.INFO)
        logging.getLogger("utils.embedder").setLevel(logging.INFO)
        logging.getLogger("utils.vectordb").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("chromadb").setLevel(logging.INFO)
        logging.getLogger("sentence_transformers").setLevel(logging.INFO)


def start_interactive_chat(chat_engine, pdf_name: str = "existing documents") -> None:
    """
    Start the interactive chat loop

    Args:
        chat_engine: Initialized chat engine
        pdf_name (str): Name of the PDF file being chatted with
    """
    print(f"\n🤖 Ready to answer questions about: {pdf_name}")
    print()

    # Enable quiet mode for chat
    set_quiet_mode(True)

    while True:
        try:
            # Get user input
            question = input("You: ").strip()

            # Check for exit commands
            if question.lower() in ["exit", "quit", "bye", "q"]:
                print("👋 Goodbye! Thanks for using Cliven!")
                break

            # Skip empty inputs
            if not question:
                continue

            # Process question
            print("🤔 Thinking...")

            # Get response from chat engine (logs are now suppressed)
            response = chat_engine.ask(question)

            # Display answer
            print(f"\n🤖 Cliven: {response['answer']}\n")

            # Show helpful tips if no context found
            if not response["context_found"]:
                print(
                    "💡 Tip: The document might not contain information about this topic.\n"
                )

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing question: {e}\n")

    # Restore normal logging when exiting
    set_quiet_mode(False)


def handle_ingest(pdf_path: str, chunk_size: int, overlap: int) -> bool:
    """
    Handle PDF ingestion with full pipeline
    """
    # Initialize console at the beginning to avoid scoping issues
    from rich.console import Console

    console = Console()

    try:
        from utils.parser import parse_pdf_with_chunking
        from utils.embedder import create_embeddings_for_chunks
        from utils.vectordb import store_embeddings_to_chromadb
        from rich.progress import Progress, SpinnerColumn, TextColumn

        # Clean and normalize the PDF path
        pdf_path_cleaned = pdf_path
        if pdf_path.startswith(("file:///", "file://")):
            pdf_path_cleaned = pdf_path.replace("file:///", "").replace("file://", "")

        # Convert to Path object and resolve
        pdf_file = Path(pdf_path_cleaned).resolve()

        console.print(f"🔍 Looking for PDF at: {pdf_file}")

        if not pdf_file.exists():
            console.print(f"❌ Error: PDF file not found: {pdf_file}", style="red")
            return False

        if not pdf_file.suffix.lower() == ".pdf":
            console.print(f"❌ Error: File must be a PDF: {pdf_file}", style="red")
            return False

        console.print(f"✅ Found PDF: {pdf_file.name}")

        # Process PDF with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Step 1: Parse and chunk PDF
            task = progress.add_task("📄 Processing PDF...", total=None)
            chunks = parse_pdf_with_chunking(
                pdf_path=str(pdf_file),
                chunk_size=chunk_size,
                overlap=overlap,
            )
            progress.update(task, description=f"✅ Created {len(chunks)} chunks")

            # Step 2: Create embeddings
            progress.update(task, description="🔄 Creating embeddings...")
            embedding_data = create_embeddings_for_chunks(chunks)
            progress.update(task, description="✅ Embeddings created")

            # Step 3: Store in ChromaDB
            progress.update(task, description="🔄 Storing in vector database...")
            # Use localhost for local development
            success = store_embeddings_to_chromadb(embedding_data, host="localhost")

            if success:
                progress.update(task, description="✅ Successfully stored in database")
                console.print(f"\n📊 Processing Summary:", style="bold green")
                console.print(f"   • File: {pdf_file.name}")
                console.print(f"   • Chunks created: {len(chunks)}")
                console.print(f"   • Chunk size: {chunk_size}")
                console.print(f"   • Overlap: {overlap}")
                console.print(
                    f"   • Embedding dimension: {embedding_data['embedding_dimension']}"
                )
                console.print(f"   • Stored in ChromaDB: ✅")
                return True
            else:
                progress.update(task, description="❌ Storage failed")
                console.print(
                    f"\n❌ Failed to store embeddings in ChromaDB", style="red"
                )
                return False

    except Exception as e:
        console.print(f"\n❌ Error processing PDF: {e}", style="red")
        logger.error(f"Error in handle_ingest: {e}")
        return False


def handle_chat_existing() -> None:
    """Handle chat with existing documents"""
    try:
        from utils.vectordb import ChromaDBManager
        from main.chat import ChatEngine

        print("🚀 Starting chat with existing documents...")
        print("=" * 50)

        # Initialize ChromaDB manager to check existing documents
        # Use localhost instead of chromadb for local development
        db_manager = ChromaDBManager(host="localhost", port=8000)

        # Check if there are existing documents
        stats = db_manager.get_collection_stats()

        if stats.get("total_chunks", 0) == 0:
            print("❌ No documents found in the database.")
            print("💡 Use 'cliven ingest <pdf_path>' to add documents first.")
            return

        print(
            f"📊 Found {stats['total_chunks']} chunks from {stats['total_documents']} documents:"
        )
        for doc_name, doc_info in stats.get("documents", {}).items():
            print(f"   • {doc_name}: {doc_info['chunk_count']} chunks")

        # Initialize chat engine
        print("\n🔄 Initializing chat engine...")
        chat_engine = ChatEngine(
            model_name="tinyllama:chat",
            chromadb_host="localhost",  # Changed from "chromadb" to "localhost"
            ollama_host="localhost",  # Changed from "ollama" to "localhost"
        )
        print("✅ Chat engine ready!")

        # Start interactive chat
        print("\n" + "=" * 50)
        print("💬 Chat with your documents! Ask any questions about the content.")
        print("Commands: 'exit', 'quit', 'bye', or 'q' to stop")
        print("=" * 50)

        start_interactive_chat(chat_engine, "existing documents")

    except Exception as e:
        logger.error(f"Error starting chat with existing docs: {e}")
        print(f"❌ Error: {e}")


def handle_chat_with_pdf(pdf_path: str) -> None:
    """Handle complete pipeline: ingest PDF and start chat"""
    try:
        from utils.parser import parse_pdf_with_chunking
        from utils.embedder import create_embeddings_for_chunks
        from utils.vectordb import store_embeddings_to_chromadb
        from main.chat import ChatEngine

        print("🚀 Starting Cliven REPL...")
        print("=" * 50)

        # Validate PDF path
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"❌ Error: PDF file not found: {pdf_path}")
            return

        if not pdf_file.suffix.lower() == ".pdf":
            print(f"❌ Error: File must be a PDF: {pdf_path}")
            return

        print(f"📄 Processing PDF: {pdf_file.name}")

        # Step 1: Parse and chunk PDF
        print("🔄 Extracting text and creating chunks...")
        chunks = parse_pdf_with_chunking(
            pdf_path=str(pdf_file), chunk_size=1000, overlap=200
        )
        print(f"✅ Created {len(chunks)} text chunks")

        # Step 2: Create embeddings
        print("🔄 Creating embeddings...")
        embedding_data = create_embeddings_for_chunks(chunks)
        print(
            f"✅ Generated embeddings (dimension: {embedding_data['embedding_dimension']})"
        )

        # Step 3: Store in ChromaDB
        print("🔄 Storing in vector database...")
        success = store_embeddings_to_chromadb(
            embedding_data, host="localhost"
        )  # Added host="localhost"
        if not success:
            print("❌ Failed to store embeddings in ChromaDB")
            return
        print("✅ Stored embeddings in ChromaDB")

        # Step 4: Initialize chat engine
        print("🔄 Initializing chat engine...")
        chat_engine = ChatEngine(
            model_name="tinyllama:chat",
            chromadb_host="localhost",  # Changed from "chromadb"
            ollama_host="localhost",  # Changed from "ollama"
        )
        print("✅ Chat engine ready!")

        # Step 5: Start interactive chat
        print("\n" + "=" * 50)
        print("💬 Chat with your PDF! Ask any questions about the content.")
        print("Commands: 'exit', 'quit', 'bye', or 'q' to stop")
        print("=" * 50)

        start_interactive_chat(chat_engine, pdf_file.name)

    except Exception as e:
        logger.error(f"Error in REPL startup: {e}")
        print(f"❌ Error: {e}")


def handle_list() -> None:
    """Handle listing documents"""
    try:
        from utils.vectordb import ChromaDBManager
        from rich.console import Console
        from rich.table import Table

        console = Console()
        db_manager = ChromaDBManager(
            host="localhost", port=8000
        )  # Changed from "chromadb"

        stats = db_manager.get_collection_stats()

        if stats.get("total_chunks", 0) == 0:
            console.print("📭 No documents found in the database.", style="yellow")
            console.print(
                "💡 Use 'cliven ingest <pdf_path>' to add documents.", style="blue"
            )
            return

        # Create table
        table = Table(title="📚 Processed Documents")
        table.add_column("Document", style="cyan", no_wrap=True)
        table.add_column("Chunks", justify="right", style="magenta")
        table.add_column("Total Size", justify="right", style="green")
        table.add_column("File Size", justify="right", style="blue")

        for doc_name, doc_info in stats.get("documents", {}).items():
            chunk_count = doc_info.get("chunk_count", 0)
            total_size = doc_info.get("total_size", 0)
            file_size = doc_info.get("file_size", 0)

            # Format sizes
            total_size_str = f"{total_size:,} chars" if total_size > 0 else "N/A"
            file_size_str = f"{file_size:,} bytes" if file_size > 0 else "N/A"

            table.add_row(doc_name, str(chunk_count), total_size_str, file_size_str)

        console.print(table)
        console.print(
            f"\n📊 Total: {stats['total_chunks']} chunks from {stats['total_documents']} documents"
        )

    except Exception as e:
        print(f"❌ Error listing documents: {e}")


def handle_status() -> None:
    """Handle system status check"""
    try:
        from main.chat import ChatEngine
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        console.print("🔍 Checking system status...", style="blue")

        # Initialize chat engine for health check
        chat_engine = ChatEngine()
        health = chat_engine.health_check()

        # Overall status
        status_color = "green" if health["overall_status"] == "healthy" else "red"
        status_text = Text(
            f"System Status: {health['overall_status'].upper()}",
            style=f"bold {status_color}",
        )

        console.print(Panel(status_text, title="🏥 Health Check"))

        # Component details
        for component, details in health.get("components", {}).items():
            component_status = details.get("status", "unknown")
            color = "green" if component_status == "healthy" else "red"

            console.print(f"\n{component.upper()}:", style="bold")
            console.print(f"  Status: {component_status}", style=color)

            if "error" in details:
                console.print(f"  Error: {details['error']}", style="red")

            if component == "ollama" and "available_models" in details:
                models = details["available_models"]
                current_model = details.get("current_model", "N/A")
                console.print(f"  Available models: {len(models)}")
                console.print(f"  Current model: {current_model}")
                if not details.get("model_available", True):
                    console.print(
                        f"  ⚠️  Warning: {current_model} not found", style="yellow"
                    )

    except Exception as e:
        print(f"❌ Error checking status: {e}")


def handle_clear(confirm: bool) -> None:
    """Handle clearing all documents"""
    try:
        from utils.vectordb import ChromaDBManager

        if not confirm:
            response = input(
                "⚠️  Are you sure you want to clear ALL documents? (yes/no): "
            )
            if response.lower() not in ["yes", "y"]:
                print("Operation cancelled.")
                return

        db_manager = ChromaDBManager(
            host="localhost", port=8000
        )  # Changed from "chromadb"
        success = db_manager.clear_collection()

        if success:
            print("✅ All documents cleared from the database.")
        else:
            print("❌ Failed to clear documents.")

    except Exception as e:
        print(f"❌ Error clearing documents: {e}")


def handle_delete(doc_id: str) -> None:
    """Handle deleting a specific document"""
    try:
        from utils.vectordb import ChromaDBManager

        db_manager = ChromaDBManager(
            host="localhost", port=8000
        )  # Changed from "chromadb"
        success = db_manager.delete_document(doc_id)

        if success:
            print(f"✅ Document '{doc_id}' deleted successfully.")
        else:
            print(f"❌ Document '{doc_id}' not found or failed to delete.")

    except Exception as e:
        print(f"❌ Error deleting document: {e}")


def handle_docker() -> None:
    """Handle Docker operations for Cliven services"""
    import subprocess
    import time
    import os
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    try:
        console.print(
            "🐳 Checking Docker status and managing Cliven services...", style="blue"
        )

        # Check if Docker daemon is running
        console.print("\n🔍 Checking Docker daemon status...")
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, "docker info")
            console.print("✅ Docker daemon is running", style="green")
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            console.print(
                "❌ Docker daemon is not running or not installed", style="red"
            )
            console.print(
                "💡 Please start Docker Desktop or Docker daemon before continuing",
                style="yellow",
            )
            console.print("   - On Windows: Start Docker Desktop")
            console.print("   - On Linux: sudo systemctl start docker")
            console.print("   - On macOS: Start Docker Desktop")
            return

        # Check if we're in the correct directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        console.print(f"📁 Working directory: {project_root}")

        # Check if docker-compose.yml exists
        if not (project_root / "docker-compose.yml").exists():
            console.print(
                "❌ docker-compose.yml not found in project root", style="red"
            )
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Step 1: Start Docker Compose services
            task = progress.add_task("🚀 Starting Docker services...", total=None)
            try:
                result = subprocess.run(
                    ["docker-compose", "up", "-d"],
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minutes timeout
                )
                if result.returncode == 0:
                    progress.update(
                        task, description="✅ Docker services started successfully"
                    )
                else:
                    progress.update(
                        task, description="❌ Failed to start Docker services"
                    )
                    console.print(f"Compose error: {result.stderr}", style="red")
                    return
            except subprocess.TimeoutExpired:
                progress.update(
                    task, description="⏰ Startup timeout - checking status..."
                )
            except Exception as e:
                progress.update(task, description=f"❌ Startup error: {str(e)}")
                return

            # Step 2: Wait for Ollama service to be ready
            task = progress.add_task("⏳ Waiting for Ollama to be ready this may take 20-30 mins depending upon connection...", total=None)
            time.sleep(10)  # Give Ollama time to start

            # Step 3: Pull the tinyllama:chat model
            task = progress.add_task("📥 Pulling tinyllama:chat model this may take 20-30 mins depending upon connection...", total=None)
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        "-it",
                        "cliven_ollama",
                        "ollama",
                        "pull",
                        "tinyllama:chat",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes timeout for model download
                )
                if result.returncode == 0:
                    progress.update(
                        task, description="✅ tinyllama:chat model pulled successfully"
                    )
                else:
                    progress.update(
                        task, description="❌ Failed to pull tinyllama:chat model"
                    )
                    console.print(f"Pull error: {result.stderr}", style="red")
                    # Don't return here, continue to show status
            except subprocess.TimeoutExpired:
                progress.update(
                    task, description="⏰ Model pull timeout - continuing anyway"
                )
            except Exception as e:
                progress.update(task, description=f"❌ Model pull error: {str(e)}")

            # Step 4: Final status check
            task = progress.add_task("🔍 Checking service status...", total=None)
            time.sleep(2)  # Brief pause before status check

        # Display service status
        console.print("\n" + "=" * 50)
        console.print("📊 Service Status Report", style="bold blue")
        console.print("=" * 50)

        # Check ChromaDB
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:8000/api/v1/heartbeat"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                console.print("🟢 ChromaDB: Running on port 8000", style="green")
            else:
                console.print("🔴 ChromaDB: Not responding", style="red")
        except:
            console.print("🔴 ChromaDB: Not accessible", style="red")

        # Check Ollama
        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/tags"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                console.print("🟢 Ollama: Running on port 11434", style="green")
                # Try to get models
                try:
                    import json

                    models_data = json.loads(result.stdout)
                    models = [model["name"] for model in models_data.get("models", [])]
                    if models:
                        console.print(
                            f"   Available models: {', '.join(models)}", style="cyan"
                        )
                        if "tinyllama:chat" in models:
                            console.print(
                                "   ✅ tinyllama:chat model is ready", style="green"
                            )
                        else:
                            console.print(
                                "   ⚠️  tinyllama:chat model not found", style="yellow"
                            )
                    else:
                        console.print("   ⚠️  No models found", style="yellow")
                except:
                    console.print(
                        "   📝 Models: Unable to parse model list", style="yellow"
                    )
            else:
                console.print("🔴 Ollama: Not responding", style="red")
        except:
            console.print("🔴 Ollama: Not accessible", style="red")

        # Check Docker containers
        console.print("\n📦 Container Status:")
        try:
            result = subprocess.run(
                ["docker-compose", "ps"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Parse and display container status
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            container_name = parts[0]
                            status = " ".join(parts[1:])
                            if "Up" in status:
                                console.print(
                                    f"   🟢 {container_name}: {status}", style="green"
                                )
                            else:
                                console.print(
                                    f"   🔴 {container_name}: {status}", style="red"
                                )
            else:
                console.print("   ❌ Unable to get container status", style="red")
        except Exception as e:
            console.print(f"   ❌ Error checking containers: {e}", style="red")

        # Provide next steps
        console.print("\n" + "=" * 50)
        console.print("🎯 Next Steps:", style="bold green")
        console.print("1. Services should now be ready for use")
        console.print("2. Run: cliven status - to check system health")
        console.print("3. Run: cliven ingest <pdf_path> - to process a PDF")
        console.print("4. Run: cliven chat - to start chatting")

        if "tinyllama:chat" not in str(result.stdout if "result" in locals() else ""):
            console.print("\n💡 If model pull failed, manually run:")
            console.print("   docker exec -it cliven_ollama ollama pull tinyllama:chat")

    except KeyboardInterrupt:
        console.print("\n⚠️  Operation cancelled by user", style="yellow")
    except Exception as e:
        console.print(f"❌ Error managing Docker services: {e}", style="red")
        console.print(
            "💡 Make sure Docker is running and you have proper permissions",
            style="blue",
        )


def handle_docker_stop() -> None:
    """Stop Docker services"""
    import subprocess
    from rich.console import Console

    console = Console()

    try:
        console.print("🛑 Stopping Cliven Docker services...", style="blue")

        # Change to project directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        result = subprocess.run(
            ["docker-compose", "down"], capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            console.print("✅ Docker services stopped successfully", style="green")
        else:
            console.print("❌ Failed to stop Docker services", style="red")
            console.print(f"Error: {result.stderr}", style="red")

    except Exception as e:
        console.print(f"❌ Error stopping Docker services: {e}", style="red")


def handle_docker_logs() -> None:
    """Show Docker service logs"""
    import subprocess
    from rich.console import Console

    console = Console()

    try:
        console.print("📋 Showing Docker service logs...", style="blue")

        # Change to project directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        # Show logs for all services
        result = subprocess.run(["docker-compose", "logs", "--tail=50"], timeout=30)

    except Exception as e:
        console.print(f"❌ Error showing logs: {e}", style="red")


def main():
    parser = argparse.ArgumentParser(
        prog="cliven",
        description="Chat with your PDF using local AI models!",
        epilog="For more information, visit: https://github.com/krey-yon/cliven",
    )

    parser.add_argument("--version", action="version", version="cliven 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Process and store PDF")
    ingest_parser.add_argument("pdf_path", help="Path to the PDF file")
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Text chunk size"
    )
    ingest_parser.add_argument(
        "--overlap", type=int, default=200, help="Chunk overlap size"
    )

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument(
        "--model", default="tinyllama:chat", help="LLM model to use"
    )
    chat_parser.add_argument(
        "--max-results", type=int, default=5, help="Max context chunks"
    )
    chat_parser.add_argument("--repl", help="Process PDF and start chat REPL")

    # List command
    list_parser = subparsers.add_parser("list", help="List processed documents")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a document")
    delete_parser.add_argument("doc_id", help="Document ID to delete")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear all documents")
    clear_parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")

    # Docker management commands
    docker_parser = subparsers.add_parser("docker", help="Manage Docker services")
    docker_subparsers = docker_parser.add_subparsers(
        dest="docker_command", help="Docker operations"
    )

    # Docker start
    docker_start_parser = docker_subparsers.add_parser(
        "start", help="Start Docker services"
    )

    # Docker stop
    docker_stop_parser = docker_subparsers.add_parser(
        "stop", help="Stop Docker services"
    )

    # Docker logs
    docker_logs_parser = docker_subparsers.add_parser(
        "logs", help="Show Docker service logs"
    )

    args = parser.parse_args()

    # If no command provided, show welcome message
    if not args.command:
        show_welcome()
        return

    try:
        if args.command == "ingest":
            success = handle_ingest(args.pdf_path, args.chunk_size, args.overlap)
            sys.exit(0 if success else 1)

        elif args.command == "chat":
            if args.repl:
                handle_chat_with_pdf(args.repl)
            else:
                handle_chat_existing()

        elif args.command == "list":
            handle_list()

        elif args.command == "delete":
            handle_delete(args.doc_id)

        elif args.command == "clear":
            handle_clear(args.confirm)

        elif args.command == "status":
            handle_status()

        elif args.command == "docker":
            if args.docker_command == "start" or args.docker_command is None:
                handle_docker()
            elif args.docker_command == "stop":
                handle_docker_stop()
            elif args.docker_command == "logs":
                handle_docker_logs()

    except ImportError as e:
        print(f"❌ Error: Missing dependency - {e}")
        print("Make sure all required packages are installed.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
