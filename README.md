# Cliven 🤖

**Chat with your PDFs using local AI models!**

Cliven is a command-line tool that allows you to process PDF documents and have interactive conversations with their content using local AI models. No data leaves your machine - everything runs locally using ChromaDB for vector storage and Ollama for AI inference.

## Features ✨

- 📄 **PDF Processing**: Extract and chunk text from PDF documents
- 🔍 **Vector Search**: Find relevant content using semantic similarity
- 🤖 **Local AI Chat**: Chat with your documents using Ollama models
- 🐳 **Docker Ready**: Easy setup with Docker Compose
- 💾 **Local Storage**: All data stays on your machine
- 🎯 **Simple CLI**: Easy-to-use command-line interface
- 🚀 **Model Selection**: Support for both lightweight (Gemma2) and high-performance (Gemma3) models
- 📊 **Rich UI**: Beautiful terminal interface with progress indicators

## Quick Start 🚀

### 1. Clone the Repository

```bash
git clone https://github.com/krey-yon/cliven.git
cd cliven
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Start Services with Docker

```bash
# Start with lightweight model (tinyllama:chat)
cliven docker start

# OR start with high-performance model (gemma3:4b)
cliven docker start --BP
# or
cliven docker start --better-performance
```

This will:

- Start ChromaDB on port 8000
- Start Ollama on port 11434
- Pull the `gemma2:2b` model (default) or `gemma3:4b` model (with --BP flag)
- May take several minutes depending on model and connection speed

### 4. Process Your First PDF

```bash
cliven ingest path/to/your/document.pdf
```

### 5. Start Chatting

```bash
# Chat with existing documents
cliven chat

# OR specify a model
cliven chat --model gemma3:2b
```

## Usage 📖

### Available Commands

```bash
# Show welcome message and commands
cliven

# Process and store a PDF
cliven ingest <pdf_path> [--chunk-size SIZE] [--overlap SIZE]

# Start interactive chat with existing documents
cliven chat [--model MODEL_NAME] [--max-results COUNT]

# Process PDF and start chat immediately
cliven chat --repl <pdf_path> [--model MODEL_NAME]

# List all processed documents
cliven list

# Delete a specific document
cliven delete <doc_id>

# Clear all documents
cliven clear [--confirm]

# Check system status
cliven status

# Manage Docker services
cliven docker start [--BP | --better-performance]  # Start services
cliven docker stop                                 # Stop services
cliven docker logs                                 # View logs
```

### Examples

```bash
# Process a manual with custom chunking
cliven ingest ./documents/user-manual.pdf --chunk-size 1500 --overlap 300

# Start chatting with all processed documents
cliven chat

# Chat with specific model
cliven chat --model gemma3:4b

# Process and chat with a specific PDF using high-performance model
cliven chat --repl ./research-paper.pdf --model gemma3:4b

# Check what documents are stored
cliven list

# Check if services are running
cliven status

# Clear all documents without confirmation
cliven clear --confirm

# Start services with better performance model
cliven docker start --BP
```

### Model Options

Cliven supports multiple AI models:

- **gemma2:2b**: Lightweight, fast responses (~1GB model)
- **gemma3:4b**: High-performance, better quality responses (~4GB model)

The system automatically selects the best available model, or you can specify one:

```bash
# Auto-select best available model
cliven chat

# Use specific model
cliven chat --model gemma3:4b
cliven chat --model gemma2:2b
```

## Architecture 🏗️

Cliven uses a modern RAG (Retrieval-Augmented Generation) architecture:

1. **PDF Parser**: Extracts text from PDFs using `pdfplumber`
2. **Text Chunker**: Splits documents into overlapping chunks using LangChain
3. **Embedder**: Creates embeddings using `BAAI/bge-small-en-v1.5`
4. **Vector Database**: Stores embeddings in ChromaDB
5. **Chat Engine**: Handles queries and generates responses with Ollama

## Components 🔧

### Core Services

- **ChromaDB**: Vector database for storing document embeddings
- **Ollama**: Local LLM inference server
- **Gemma2:2b**: Lightweight chat model for fast responses
- **Gemma3:4b**: High-performance model for better quality responses

### Key Files

- `main/cliven.py`: Main CLI application with argument parsing
- `main/chat.py`: Chat engine with RAG functionality and model management
- `utils/parser.py`: PDF text extraction and chunking
- `utils/embedder.py`: Text embedding generation using sentence transformers
- `utils/vectordb.py`: ChromaDB operations and vector storage
- `utils/chunker.py`: Text chunking utilities
- `docker-compose.yml`: Service orchestration configuration

## System Requirements 📋

### Software Requirements

- Python 3.8+
- Docker & Docker Compose
- 2GB+ RAM (for TinyLlama model)
- 8GB+ RAM (for Gemma3 4B model)
- 4GB+ disk space

### Python Dependencies

- `typer>=0.9.0` - CLI framework
- `rich>=13.0.0` - Beautiful terminal output
- `pdfplumber>=0.7.0` - PDF text extraction
- `sentence-transformers>=2.2.0` - Text embeddings
- `chromadb>=0.4.0` - Vector database
- `langchain>=0.0.300` - Text processing
- `requests>=2.28.0` - HTTP client

## Installation Options 🛠️

### Option 1: Local Development

```bash
# Clone repository
git clone https://github.com/krey-yon/cliven.git
cd cliven

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -e .

# Start services
cliven docker start
```

### Option 2: Production Install

```bash
pip install git+https://github.com/krey-yon/cliven.git
```

## Configuration ⚙️

### Environment Variables

```bash
# ChromaDB settings
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Ollama settings
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

### Customization

```bash
# Use different chunk sizes
cliven ingest document.pdf --chunk-size 1500 --overlap 300

# Use different model
cliven chat --model gemma3:4b

# Adjust context window
cliven chat --max-results 10

# Skip confirmation for clearing
cliven clear --confirm
```

### Model Management

```bash
# Check available models
cliven status

# Manually pull models
docker exec -it cliven_ollama ollama pull gemma3:4b
docker exec -it cliven_ollama ollama pull gemma2:2b

# List downloaded models
docker exec -it cliven_ollama ollama list
```

## Troubleshooting 🔧

### Common Issues

1. **Docker services not starting**

   ```bash
   # Check Docker daemon
   docker info

   # View service logs
   cliven docker logs

   # Restart services
   cliven docker stop
   cliven docker start
   ```

2. **Model not found**

   ```bash
   # Check available models
   cliven status

   # Manually pull model
   docker exec -it cliven_ollama ollama pull gemma3:4b
   docker exec -it cliven_ollama ollama pull tinyllama:chat
   ```

3. **ChromaDB connection failed**

   ```bash
   # Check service status
   cliven status

   # Restart services
   cliven docker stop
   cliven docker start

   # Check logs
   cliven docker logs
   ```

4. **PDF processing errors**

   ```bash
   # Check file path and permissions
   dir path\to\file.pdf

   # Try with different chunk size
   cliven ingest file.pdf --chunk-size 500

   # Check for PDF corruption
   cliven ingest file.pdf --chunk-size 2000 --overlap 100
   ```

5. **Model performance issues**

   ```bash
   # Switch to lightweight model
   cliven chat --model gemma2:2b

   # Or use high-performance model
   cliven chat --model gemma3:4b

   # Check system resources
   cliven status
   ```

### Performance Tips

- Use `gemma2:2b` for faster responses on limited hardware
- Use `gemma3:4b` for better quality responses with sufficient RAM
- Use smaller chunk sizes for better context precision
- Increase overlap for better continuity
- Monitor RAM usage with large PDFs
- Use SSD storage for better ChromaDB performance


## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Ollama](https://ollama.ai/) for local LLM inference
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [LangChain](https://langchain.com/) for text processing
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- [PDFplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction

## Support 💬

- 📧 Email: vikaskumar783588@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/krey-yon/cliven/issues)
- 💡 Discussions: [GitHub Discussions](https://github.com/krey-yon/cliven/discussions)

---

**Made with ❤️ by [Kreyon](https://github.com/krey-yon)**

_Chat with your PDFs locally and securely!_
