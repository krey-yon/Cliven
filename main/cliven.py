import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def show_welcome():
    """Display welcome message and available commands"""
    print(
        r"""
┌───────────────────────────────────────────────────────────────────────────────┐
│ 🤖  ██████╗██╗     ██╗██╗   ██╗███████╗███╗   ██╗                             │
│     ██╔════╝██║     ██║██║   ██║██╔════╝████╗  ██║                            │
│     ██║     ██║     ██║██║   ██║█████╗  ██╔██╗ ██║  Cliven - Chat with PDFs   │
│     ██║     ██║     ██║██║   ██║██╔══╝  ██║╚██╗██║  CLI Tool                  │
│     ╚██████╗███████╗██║╚██████╔╝███████╗██║ ╚████║                            │
│      ╚═════╝╚══════╝╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝                            │
├───────────────────────────────────────────────────────────────────────────────┤
│ 🛠️  Available Commands:                                                       │
│                                                                               │
│  cliven ingest <pdf_path>           → Process and store PDF in vector DB      │
│     └─ --chunk-size SIZE            → Set chunk size (default: 1000)          │
│     └─ --overlap SIZE               → Set chunk overlap (default: 200)        │
│                                                                               │
│  cliven chat                         → Start interactive chat with documents  │
│     └─ --model MODEL_NAME           → Use specific LLM model                  │
│     └─ --max-results COUNT          → Context chunk limit (default: 5)        │
│     └─ --repl <pdf_path>            → Ingest & chat immediately               │
│                                                                               │
│  cliven list                         → List all processed documents           │
│  cliven delete <doc_id>              → Delete a processed document            │
│  cliven clear                        → Clear all documents                    │
│     └─ --confirm                    → Skip confirmation prompt                │
│                                                                               │
│  cliven status                       → Show system status (ChromaDB, Ollama)  │
│                                                                               │
│  cliven docker start                 → Start Docker services                  │
│     └─ --BP / --better-performance → Use gemma3:4b for higher performance     │
│  cliven docker stop                  → Stop Docker services                   │
│  cliven docker logs                  → Show Docker logs                       │
│                                                                               │
│  cliven --help                       → Show detailed help                     │
│  cliven --version                    → Show version information               │
├───────────────────────────────────────────────────────────────────────────────┤
│ 📌 Examples:                                                                  │
│                                                                               │
│  cliven ingest ./docs/manual.pdf --chunk-size 1500 --overlap 300              │
│  cliven chat --model gemma3:4b                                                │
│  cliven chat --repl ./docs/manual.pdf --model gemma3:4b                       │
│  cliven list                                                                  │
│  cliven delete manual.pdf                                                     │
│  cliven clear --confirm                                                       │
│  cliven status                                                                │
│  cliven docker start --BP                                                     │
│  cliven docker logs                                                           │
├───────────────────────────────────────────────────────────────────────────────┤
│ ✅ Cliven is ready! Use any command above to get started.                     │
└───────────────────────────────────────────────────────────────────────────────┘
"""
    )



def start_interactive_chat(chat_engine, pdf_name: str = "existing documents") -> None:
    """
    Start the interactive chat loop

    Args:
        chat_engine: Initialized chat engine
        pdf_name (str): Name of the PDF file being chatted with
    """
    print(f"\n🤖 Ready to answer questions about: {pdf_name}")
    print()

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

            # Get response from chat engine
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
        return False


def get_available_models() -> List[str]:
    """Get list of available models from Ollama"""
    import subprocess

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


def select_best_available_model() -> str:
    """Select the best available model based on priority"""
    available_models = get_available_models()

    # Priority order: gemma3:4b first, then gemma2:2b, then any available
    priority_models = ["gemma3:4b", "gemma2:2b"]

    for model in priority_models:
        if model in available_models:
            return model

    # If none of the priority models are available, return first available or default
    if available_models:
        return available_models[0]

    return "gemma2:2b"  # Default fallback


def handle_docker(use_better_model: bool = False) -> None:
    """Handle Docker operations for Cliven services"""
    import subprocess
    import time
    import os
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    # Determine which model to pull
    model_to_pull = "gemma3:4b" if use_better_model else "gemma2:2b"
    model_description = (
        "better performance model" if use_better_model else "lightweight model"
    )

    try:
        console.print(
            f"🐳 Checking Docker status and managing Cliven services with {model_description}...",
            style="blue",
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
            task = progress.add_task(
                "⏳ Waiting for Ollama to be ready this may take several mins depending upon connection...",
                total=None,
            )
            time.sleep(10)  # Give Ollama time to start

            # Step 3: Pull the specified model
            task = progress.add_task(
                f"📥 Pulling {model_to_pull} model this may take several mins depending upon connection...",
                total=None,
            )
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        "-it",
                        "cliven_ollama",
                        "ollama",
                        "pull",
                        model_to_pull,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=800,
                )
                if result.returncode == 0:
                    progress.update(
                        task,
                        description=f"✅ {model_to_pull} model pulled successfully",
                    )
                else:
                    progress.update(
                        task, description=f"❌ Failed to pull {model_to_pull} model"
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

        # Check Ollama and list available models
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

                        # Check for specific models
                        if "gemma2:2b" in models:
                            console.print(
                                "   ✅ gemma2:2b model is ready", style="green"
                            )
                        if "gemma3:4b" in models:
                            console.print(
                                "   ✅ gemma3:4b model is ready", style="green"
                            )

                        # Show which model was just pulled
                        if model_to_pull in models:
                            console.print(
                                f"   🎯 {model_to_pull} is now available for chat",
                                style="bold green",
                            )
                        else:
                            console.print(
                                f"   ⚠️  {model_to_pull} model not found after pull",
                                style="yellow",
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

        if use_better_model:
            console.print(
                "5. Chat will automatically use gemma3:4b for better responses"
            )
        else:
            console.print("5. Chat will use gemma2:2b for faster responses")
            console.print(
                "   💡 Use 'cliven docker start --BP' for better model (gemma3:4b)"
            )

        # Show manual command if model pull failed
        available_models = get_available_models()
        if model_to_pull not in available_models:
            console.print(f"\n💡 If model pull failed, manually run:")
            console.print(
                f"   docker exec -it cliven_ollama ollama pull {model_to_pull}"
            )

    except KeyboardInterrupt:
        console.print("\n⚠️  Operation cancelled by user", style="yellow")
    except Exception as e:
        console.print(f"❌ Error managing Docker services: {e}", style="red")
        console.print(
            "💡 Make sure Docker is running and you have proper permissions",
            style="blue",
        )


def handle_chat_existing(model_override: Optional[str] = None) -> None:
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

        # Use model override if provided, otherwise auto-select
        if model_override:
            selected_model = model_override
            print(f"\n🤖 Using specified model: {selected_model}")
        else:
            selected_model = select_best_available_model()
            print(f"\n🤖 Auto-selected model: {selected_model}")

        if selected_model == "gemma3:4b":
            print("   🚀 Using high-performance model for better responses")
        elif selected_model == "gemma2:2b":
            print("   ⚡ Using lightweight model for faster responses")
        else:
            print(f"   📝 Using available model: {selected_model}")

        # Initialize chat engine
        print("\n🔄 Initializing chat engine...")
        chat_engine = ChatEngine(
            model_name=selected_model,
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
        print(f"❌ Error: {e}")


def handle_chat_with_pdf(pdf_path: str, model_override: Optional[str] = None) -> None:
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

        # Step 4: Use model override if provided, otherwise auto-select
        if model_override:
            selected_model = model_override
            print(f"\n🤖 Using specified model: {selected_model}")
        else:
            selected_model = select_best_available_model()
            print(f"\n🤖 Auto-selected model: {selected_model}")

        if selected_model == "gemma3:4b":
            print("   🚀 Using high-performance model for better responses")
        elif selected_model == "gemma2:2b":
            print("   ⚡ Using lightweight model for faster responses")
        else:
            print(f"   📝 Using available model: {selected_model}")

        # Step 5: Initialize chat engine
        print("🔄 Initializing chat engine...")
        chat_engine = ChatEngine(
            model_name=selected_model,
            chromadb_host="localhost",  # Changed from "chromadb"
            ollama_host="localhost",  # Changed from "ollama"
        )
        print("✅ Chat engine ready!")

        # Step 6: Start interactive chat
        print("\n" + "=" * 50)
        print("💬 Chat with your PDF! Ask any questions about the content.")
        print("Commands: 'exit', 'quit', 'bye', or 'q' to stop")
        print("=" * 50)

        start_interactive_chat(chat_engine, pdf_file.name)

    except Exception as e:
        print(f"❌ Error: {e}")


def handle_list() -> None:
    """Handle listing all processed documents"""
    try:
        from utils.vectordb import ChromaDBManager
        from rich.console import Console
        from rich.table import Table

        console = Console()

        console.print("📋 Listing processed documents...", style="blue")

        # Initialize ChromaDB manager
        db_manager = ChromaDBManager(host="localhost", port=8000)

        # Get collection stats
        stats = db_manager.get_collection_stats()

        if stats.get("total_chunks", 0) == 0:
            console.print("📄 No documents found in the database.", style="yellow")
            console.print("💡 Use 'cliven ingest <pdf_path>' to add documents first.")
            return

        # Create a table for better display
        table = Table(title="📚 Processed Documents")
        table.add_column("Document Name", style="cyan")
        table.add_column("Chunks", justify="right", style="green")
        table.add_column("Status", style="bold")

        # Add documents to table
        for doc_name, doc_info in stats.get("documents", {}).items():
            chunk_count = doc_info.get("chunk_count", 0)
            status = "✅ Ready" if chunk_count > 0 else "⚠️ Empty"
            table.add_row(doc_name, str(chunk_count), status)

        console.print(table)

        # Summary
        console.print(f"\n📊 Summary:", style="bold")
        console.print(f"   • Total documents: {stats.get('total_documents', 0)}")
        console.print(f"   • Total chunks: {stats.get('total_chunks', 0)}")
        console.print(f"   • Database status: ✅ Connected")

    except Exception as e:
        console.print(f"❌ Error listing documents: {e}", style="red")


def handle_delete(doc_id: str) -> None:
    """Handle deleting a specific document"""
    try:
        from utils.vectordb import ChromaDBManager
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()

        console.print(f"🗑️  Preparing to delete document: {doc_id}", style="yellow")

        # Initialize ChromaDB manager
        db_manager = ChromaDBManager(host="localhost", port=8000)

        # Check if document exists
        stats = db_manager.get_collection_stats()
        documents = stats.get("documents", {})

        if doc_id not in documents:
            console.print(f"❌ Document '{doc_id}' not found.", style="red")
            console.print("Available documents:")
            for doc_name in documents.keys():
                console.print(f"   • {doc_name}")
            return

        # Get document info
        doc_info = documents[doc_id]
        chunk_count = doc_info.get("chunk_count", 0)

        console.print(f"📄 Document: {doc_id}")
        console.print(f"📊 Chunks to delete: {chunk_count}")

        # Confirm deletion
        if not Confirm.ask(f"Are you sure you want to delete '{doc_id}'?"):
            console.print("❌ Deletion cancelled.", style="yellow")
            return

        # Delete document
        success = db_manager.delete_document(doc_id)

        if success:
            console.print(
                f"✅ Successfully deleted '{doc_id}' and its {chunk_count} chunks.",
                style="green",
            )
        else:
            console.print(f"❌ Failed to delete '{doc_id}'.", style="red")

    except Exception as e:
        console.print(f"❌ Error deleting document: {e}", style="red")


def handle_clear(skip_confirmation: bool = False) -> None:
    """Handle clearing all documents"""
    try:
        from utils.vectordb import ChromaDBManager
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()

        console.print("🗑️  Preparing to clear all documents...", style="yellow")

        # Initialize ChromaDB manager
        db_manager = ChromaDBManager(host="localhost", port=8000)

        # Get current stats
        stats = db_manager.get_collection_stats()
        total_docs = stats.get("total_documents", 0)
        total_chunks = stats.get("total_chunks", 0)

        if total_docs == 0:
            console.print(
                "📄 No documents to clear. Database is already empty.", style="blue"
            )
            return

        console.print(f"📊 Current database contents:")
        console.print(f"   • Documents: {total_docs}")
        console.print(f"   • Chunks: {total_chunks}")

        # Confirm deletion unless skipped
        if not skip_confirmation:
            console.print(
                "\n⚠️  This will permanently delete ALL processed documents!",
                style="red",
            )
            if not Confirm.ask("Are you sure you want to clear the entire database?"):
                console.print("❌ Clear operation cancelled.", style="yellow")
                return

        # Clear database
        console.print("🔄 Clearing database...")
        success = db_manager.clear_collection()

        if success:
            console.print(
                f"✅ Successfully cleared {total_docs} documents and {total_chunks} chunks.",
                style="green",
            )
            console.print("💡 Use 'cliven ingest <pdf_path>' to add new documents.")
        else:
            console.print("❌ Failed to clear database.", style="red")

    except Exception as e:
        console.print(f"❌ Error clearing database: {e}", style="red")


def handle_status() -> None:
    """Handle showing system status"""
    try:
        import subprocess
        import requests
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        console.print("🔍 Checking Cliven system status...", style="blue")

        # Create status table
        table = Table(title="🖥️  Cliven System Status")
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")

        # Check Docker daemon
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                table.add_row("Docker Daemon", "🟢 Running", "Docker is available")
            else:
                table.add_row("Docker Daemon", "🔴 Stopped", "Docker not running")
        except:
            table.add_row(
                "Docker Daemon", "🔴 Error", "Docker not installed/accessible"
            )

        # Check Docker containers
        try:
            result = subprocess.run(
                ["docker-compose", "ps"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                chromadb_running = False
                ollama_running = False

                for line in lines:
                    if line.strip():
                        if "cliven_chromadb" in line and "Up" in line:
                            chromadb_running = True
                        if "cliven_ollama" in line and "Up" in line:
                            ollama_running = True

                if chromadb_running:
                    table.add_row(
                        "ChromaDB Container", "🟢 Running", "Vector database active"
                    )
                else:
                    table.add_row(
                        "ChromaDB Container", "🔴 Stopped", "Container not running"
                    )

                if ollama_running:
                    table.add_row(
                        "Ollama Container", "🟢 Running", "AI model server active"
                    )
                else:
                    table.add_row(
                        "Ollama Container", "🔴 Stopped", "Container not running"
                    )
            else:
                table.add_row(
                    "Containers", "🔴 Error", "Could not check container status"
                )
        except:
            table.add_row("Containers", "🔴 Error", "Docker Compose not available")

        # Check ChromaDB API
        try:
            response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=5)
            if response.status_code == 200:
                table.add_row(
                    "ChromaDB API", "🟢 Healthy", "Port 8000 - API responding"
                )
            else:
                table.add_row(
                    "ChromaDB API", "🔴 Unhealthy", f"HTTP {response.status_code}"
                )
        except:
            table.add_row("ChromaDB API", "🔴 Down", "Port 8000 - Not accessible")

        # Check Ollama API and models
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                table.add_row("Ollama API", "🟢 Healthy", "Port 11434 - API responding")

                # Check available models
                models_data = response.json()
                models = [model["name"] for model in models_data.get("models", [])]
                if models:
                    model_list = ", ".join(models[:3])  # Show first 3 models
                    if len(models) > 3:
                        model_list += f" (+{len(models)-3} more)"
                    table.add_row("AI Models", "🟢 Available", model_list)
                else:
                    table.add_row("AI Models", "🔴 None", "No models downloaded")
            else:
                table.add_row(
                    "Ollama API", "🔴 Unhealthy", f"HTTP {response.status_code}"
                )
        except:
            table.add_row("Ollama API", "🔴 Down", "Port 11434 - Not accessible")

        # Check database contents
        try:
            from utils.vectordb import ChromaDBManager

            db_manager = ChromaDBManager(host="localhost", port=8000)
            stats = db_manager.get_collection_stats()

            total_docs = stats.get("total_documents", 0)
            total_chunks = stats.get("total_chunks", 0)

            if total_docs > 0:
                table.add_row(
                    "Document Database",
                    "🟢 Ready",
                    f"{total_docs} docs, {total_chunks} chunks",
                )
            else:
                table.add_row("Document Database", "🔴 Empty", "No documents processed")
        except:
            table.add_row("Document Database", "🔴 Error", "Cannot access database")

        console.print(table)

        # System recommendations
        console.print("\n💡 System Recommendations:", style="bold yellow")

        # Check if services need to be started
        try:
            docker_running = (
                subprocess.run(["docker", "info"], capture_output=True).returncode == 0
            )
            if not docker_running:
                console.print("   • Start Docker Desktop or Docker daemon")
            else:
                containers_result = subprocess.run(
                    ["docker-compose", "ps"], capture_output=True, text=True
                )
                if "Up" not in containers_result.stdout:
                    console.print("   • Run: cliven docker start")
        except:
            console.print("   • Install Docker and Docker Compose")

        # Check if models need to be downloaded
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                if not models:
                    console.print(
                        "   • Run: cliven docker start (to download AI models)"
                    )
        except:
            pass

        # Check if documents need to be added
        try:
            from utils.vectordb import ChromaDBManager

            db_manager = ChromaDBManager(host="localhost", port=8000)
            stats = db_manager.get_collection_stats()
            if stats.get("total_documents", 0) == 0:
                console.print("   • Run: cliven ingest <pdf_path> (to add documents)")
        except:
            pass

    except Exception as e:
        console.print(f"❌ Error checking system status: {e}", style="red")


def handle_docker_stop() -> None:
    """Handle stopping Docker services"""
    try:
        import subprocess
        import os
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn

        console = Console()

        console.print("🛑 Stopping Cliven Docker services...", style="yellow")

        # Change to project directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            task = progress.add_task("🔄 Stopping containers...", total=None)

            try:
                result = subprocess.run(
                    ["docker-compose", "down"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    progress.update(
                        task, description="✅ Services stopped successfully"
                    )
                    console.print(
                        "\n🟢 All Cliven services have been stopped.", style="green"
                    )
                    console.print(
                        "💡 Use 'cliven docker start' to restart services when needed."
                    )
                else:
                    progress.update(task, description="❌ Failed to stop services")
                    console.print(
                        f"\n❌ Error stopping services: {result.stderr}", style="red"
                    )

            except subprocess.TimeoutExpired:
                progress.update(task, description="⏰ Stop operation timed out")
                console.print(
                    "\n⚠️  Stop operation timed out, but services may still be stopping.",
                    style="yellow",
                )
            except Exception as e:
                progress.update(task, description=f"❌ Error: {str(e)}")
                console.print(f"\n❌ Error: {e}", style="red")

        # Show final container status
        console.print("\n📦 Final container status:")
        try:
            result = subprocess.run(
                ["docker-compose", "ps"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                if not lines or not any(line.strip() for line in lines):
                    console.print("   📴 No containers running", style="green")
                else:
                    for line in lines:
                        if line.strip():
                            console.print(f"   {line}")
            else:
                console.print("   ❌ Could not check container status", style="red")
        except:
            console.print("   ❌ Error checking container status", style="red")

    except Exception as e:
        console.print(f"❌ Error stopping Docker services: {e}", style="red")


def handle_docker_logs() -> None:
    """Handle showing Docker service logs"""
    try:
        import subprocess
        import os
        from rich.console import Console
        from rich.prompt import Prompt
        from rich.panel import Panel

        console = Console()

        # Change to project directory
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        console.print("📋 Cliven Docker Service Logs", style="bold blue")

        # Ask which service to show logs for
        console.print("\nAvailable services:")
        console.print("   1. chromadb  - Vector database logs")
        console.print("   2. ollama    - AI model server logs")
        console.print("   3. all       - All service logs")

        choice = Prompt.ask(
            "\nWhich service logs would you like to see?",
            choices=["1", "2", "3", "chromadb", "ollama", "all"],
            default="all",
        )

        # Map choices to service names
        service_map = {
            "1": "chromadb",
            "2": "ollama",
            "3": "all",
            "chromadb": "chromadb",
            "ollama": "ollama",
            "all": "all",
        }

        selected_service = service_map.get(choice, "all")

        console.print(f"\n📋 Showing logs for: {selected_service}")
        console.print("Press Ctrl+C to exit log viewer\n")

        try:
            if selected_service == "all":
                # Show all service logs
                result = subprocess.run(
                    ["docker-compose", "logs", "--tail=100", "-f"],
                    timeout=None,  # No timeout for log following
                )
            else:
                # Show specific service logs
                service_name = f"cliven_{selected_service}"
                result = subprocess.run(
                    ["docker-compose", "logs", "--tail=100", "-f", selected_service],
                    timeout=None,  # No timeout for log following
                )

        except KeyboardInterrupt:
            console.print("\n\n👋 Log viewer stopped.", style="yellow")
        except subprocess.CalledProcessError as e:
            console.print(f"\n❌ Error accessing logs: {e}", style="red")
            console.print(
                "💡 Make sure Docker services are running: cliven docker start"
            )
        except Exception as e:
            console.print(f"\n❌ Unexpected error: {e}", style="red")

    except Exception as e:
        console.print(f"❌ Error showing Docker logs: {e}", style="red")


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
        "--model", default=None, help="LLM model to use (overrides auto-selection)"
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
    docker_start_parser.add_argument(
        "--BP",
        "--better-performance",
        action="store_true",
        help="Use gemma3:4b model for better performance instead of gemma2:2b",
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
                handle_chat_with_pdf(args.repl, model_override=args.model)
            else:
                handle_chat_existing(model_override=args.model)

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
                use_better_model = getattr(args, "BP", False)
                handle_docker(use_better_model)
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
