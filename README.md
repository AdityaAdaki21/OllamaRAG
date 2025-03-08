# OllamaRAG

A local PDF Question-Answering system using Ollama and ChromaDB for Retrieval Augmented Generation (RAG).

## Overview

OllamaRAG is a powerful, locally-run tool that allows you to query PDF documents using Large Language Models. The system leverages Ollama for running LLMs and embeddings locally, and ChromaDB for efficient vector storage and retrieval.

Key features:
- 🚀 100% local execution with no data sent to external services
- 📄 PDF ingestion with automatic chunking and embedding
- 🔍 Semantic search to find relevant context
- 💡 Context-aware responses from the LLM
- 🧠 Persistent knowledge base with ChromaDB
- 🌈 Beautiful terminal UI with rich formatting

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running
- Python 3.9 with pip
- Required models pulled in Ollama:
  - An LLM model (default: `tinyllama`)
  - An embedding model (default: `mxbai-embed-large`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ollamarag.git
cd ollamarag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running:
```bash
ollama serve
```

4. Pull the required models:
```bash
ollama pull tinyllama
ollama pull mxbai-embed-large
```

## Usage

### Ingesting a PDF

```bash
python ollamarag.py --pdf path/to/your/document.pdf
```

Optional parameters:
- `--llm` - LLM model to use (default: `tinyllama`)
- `--embedder` - Embedding model to use (default: `mxbai-embed-large`)
- `--workers` - Number of parallel workers for processing (default: `4`)
- `--context-window` - Context window size for the LLM (default: `2048`)

### Using an Existing Collection

```bash
python ollamarag.py --collection collection_name
```

### Interactive Commands

Once the application is running, you can use the following commands:
- Just type your question to query the document
- `exit`, `quit` - End the session
- `list` - List available collections in ChromaDB
- `use <collection>` - Switch to a different collection
- `help` - Display help information
- `stats` - Display knowledge base statistics
- `clear` - Clear the terminal screen

## How It Works

1. **PDF Ingestion**: The PDF is converted to text, chunked into smaller segments, and stored in ChromaDB.
2. **Embedding Generation**: Each text chunk is transformed into a vector embedding using the specified embedding model.
3. **Semantic Search**: When you ask a question, the system finds the most relevant chunks by embedding similarity.
4. **Context-Aware Response**: The LLM generates a response based on your question and the retrieved context.

## Advanced Configuration

### Models

You can use any model available in Ollama:

```bash
# Use a different LLM model
python ollamarag.py --pdf document.pdf --llm llama3.2

# Use a different embedding model
python ollamarag.py --pdf document.pdf --embedder nomic-embed-text
```

### Performance Tuning

Adjust workers and chunking parameters for better performance:

```bash
# Use more parallel workers for faster processing
python ollamarag.py --pdf document.pdf --workers 8
```

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running with `ollama serve`
- **Missing Models**: If you see model warnings, pull them with `ollama pull <model_name>`
- **Memory Issues**: Try using a smaller model like `tinyllama` or reducing the number of workers

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing local LLM execution
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [PyPDF2](https://pypdf2.readthedocs.io/) for PDF processing
- [Rich](https://rich.readthedocs.io/) for terminal UI