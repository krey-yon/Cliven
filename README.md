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
cliven docker start
```

This will:
- Start ChromaDB on port 8000
- Start Ollama on port 11434
- Pull the `tinyllama:chat` model (may take 20-30 minutes)

### 4. Process Your First PDF

```bash
cliven ingest path/to/your/document.pdf
```

### 5. Start Chatting

```bash
cliven chat
```

## Usage 📖

### Available Commands

```bash
# Show welcome message and commands
cliven

# Process and store a PDF
cliven ingest <pdf_path>

# Start interactive chat with existing documents
cliven chat

# Process PDF and start chat immediately
cliven chat --repl <pdf_path>

# List all processed documents
cliven list

# Delete a specific document
cliven delete <doc_id>

# Clear all documents
cliven clear

# Check system status
cliven status

# Manage Docker services
cliven docker start    # Start services
cliven docker stop     # Stop services
cliven docker logs     # View logs
```

### Examples

```bash
# Process a manual
cliven ingest ./documents/user-manual.pdf

# Start chatting with all processed documents
cliven chat

# Process and chat with a specific PDF
cliven chat --repl ./research-paper.pdf

# Check what documents are stored
cliven list

# Check if services are running
cliven status
```

## Architecture 🏗️

Cliven uses a modern RAG (Retrieval-Augmented Generation) architecture:

1. **PDF Parser**: Extracts text from PDFs using `pdfplumber`
2. **Text Chunker**: Splits documents into overlapping chunks using LangChain
3. **Embedder**: Creates embeddings using `BAAI/bge-small-en-v1.5`
4. **Vector Database**: Stores embeddings in ChromaDB
5. **Chat Engine**: Handles queries and generates responses

## Components 🔧

### Core Services

- **ChromaDB**: Vector database for storing document embeddings
- **Ollama**: Local LLM inference server
- **TinyLlama**: Lightweight chat model for responses

### Key Files

- `main/cliven.py`: Main CLI application
- `main/chat.py`: Chat engine with RAG functionality
- `utils/parser.py`: PDF text extraction
- `utils/embedder.py`: Text embedding generation
- `utils/vectordb.py`: ChromaDB operations
- `docker-compose.yml`: Service orchestration

## System Requirements 📋

### Software Requirements

- Python 3.8+
- Docker & Docker Compose
- 4GB+ RAM (for TinyLlama model)
- 2GB+ disk space

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
cliven chat --model llama2:chat

# Adjust context window
cliven chat --max-results 10
```

## Troubleshooting 🔧

### Common Issues

1. **Docker services not starting**
   ```bash
   # Check Docker daemon
   docker info
   
   # View service logs
   cliven docker logs
   ```

2. **Model not found**
   ```bash
   # Manually pull model
   docker exec -it cliven_ollama ollama pull tinyllama:chat
   ```

3. **ChromaDB connection failed**
   ```bash
   # Check service status
   cliven status
   
   # Restart services
   cliven docker stop
   cliven docker start
   ```

4. **PDF processing errors**
   ```bash
   # Check file path and permissions
   dir path\to\file.pdf
   
   # Try with different chunk size
   cliven ingest file.pdf --chunk-size 500
   ```

### Performance Tips

- Use smaller chunk sizes for better context precision
- Increase overlap for better continuity
- Monitor RAM usage with large PDFs
- Use SSD storage for better ChromaDB performance

## Development 👨‍💻

### Setup Development Environment

```bash
git clone https://github.com/krey-yon/cliven.git
cd cliven

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black .
isort .
flake8 .
```

### Project Structure

```
cliven/
├── main/
│   ├── __init__.py
│   ├── cliven.py      # Main CLI application
│   └── chat.py        # Chat engine
├── utils/
│   ├── parser.py      # PDF processing
│   ├── embedder.py    # Text embeddings
│   ├── vectordb.py    # ChromaDB operations
│   └── chunker.py     # Text chunking
├── docker-compose.yml # Service configuration
├── setup.py          # Package configuration
└── requirements.txt   # Dependencies
```

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

## Support 💬

- 📧 Email: vikaskumar783588@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/krey-yon/cliven/issues)
- 💡 Discussions: [GitHub Discussions](https://github.com/krey-yon/cliven/discussions)

---

**Made with ❤️ by [Kreyon](https://github.com/krey-yon)**

*Chat with your PDFs locally and securely!*