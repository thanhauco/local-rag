# Local RAG System

A complete Retrieval-Augmented Generation (RAG) system built from scratch on macOS using Python. No cloud platforms, no IDE "magic" — every stage is explicit and visible.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│  Documents  │───▶│ Text Parsing │───▶│   Chunking   │───▶│ Embeddings │
│  (PDF/TXT)  │    │   PyPDF      │    │ Recursive    │    │ MiniLM-L6  │
└─────────────┘    └──────────────┘    └──────────────┘    └────────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│   Answer    │◀───│  Generation  │◀───│  Retrieval   │◀───│  Pinecone  │
│             │    │  FLAN-T5     │    │ RetrievalQA  │    │   Vector   │
└─────────────┘    └──────────────┘    └──────────────┘    └────────────┘
```

## Features

- **Document Ingestion**: Load PDF and TXT files using LangChain's DirectoryLoader and PyPDFLoader
- **Text Chunking**: RecursiveCharacterTextSplitter with configurable chunk size and overlap
- **Embeddings**: HuggingFace MiniLM-L6-v2 (384 dimensions, runs locally)
- **Vector Storage**: Pinecone for efficient similarity search
- **Retrieval**: LangChain RetrievalQA chain for context-aware querying
- **Generation**: FLAN-T5-Base instruction-tuned LLM (runs on CPU)

## Installation

```bash
# Clone the repository
git clone https://github.com/thanhauco/local-rag-system.git
cd local-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Pinecone API key
```

## Configuration

Create a `.env` file with:

```env
PINECONE_API_KEY=your-api-key
PINECONE_INDEX_NAME=local-rag-index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
```

## Usage

### Ingest Documents

```bash
# Add documents to data/documents/
python main.py ingest ./data/documents
```

### Query the System

```bash
# Single query
python main.py query "What is the main topic of these documents?"

# Interactive mode
python main.py interactive
```

## Project Structure

```
local-rag-system/
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── document_loader.py  # PDF/TXT ingestion
│   ├── text_processor.py   # Text chunking
│   ├── embeddings.py       # HuggingFace embeddings
│   ├── vector_store.py     # Pinecone integration
│   ├── retrieval.py        # RetrievalQA chain
│   └── generator.py        # FLAN-T5 generation
├── data/
│   └── documents/          # Your documents here
├── tests/
├── main.py                 # CLI entry point
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Tech Stack

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| Orchestration    | LangChain                      |
| Document Loading | PyPDFLoader, DirectoryLoader   |
| Text Splitting   | RecursiveCharacterTextSplitter |
| Embeddings       | HuggingFace MiniLM-L6-v2       |
| Vector Store     | Pinecone                       |
| Retrieval        | LangChain RetrievalQA          |
| Generation       | FLAN-T5-Base                   |

## Why Build From Scratch?

When you understand how RAG works end-to-end, you can design systems that are:

- **Cloud-portable**: Not locked to any vendor
- **Cost-aware**: Understand resource consumption
- **Governable**: Full control over data flow
- **Resilient**: No dependency on SDK changes

## License

MIT License - See LICENSE file for details.

## Author

**Tha Vu**
