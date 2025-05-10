# Local RAG System - Implementation Plan

Build a complete Retrieval-Augmented Generation system from scratch on macOS using Python.

## Architecture Overview

```
Document Input → Text Parsing → Text Chunking → Embeddings → Vector Storage → Retrieval → Generation
     │              │               │              │              │             │            │
   PDF/TXT    DirectoryLoader  Recursive      MiniLM-L6-v2    Pinecone    RetrievalQA   FLAN-T5
              PyPDFLoader      TextSplitter
```

## Implementation Phases

### Phase 1: Project Setup

- [x] Initialize git repository
- [x] Create .gitignore
- [x] Create README.md
- [x] Create implementation_plan.md
- [x] Create requirements.txt
- [x] Create configuration module

### Phase 2: Document Ingestion

- [x] Implement PDF loader (PyPDFLoader)
- [x] Implement TXT loader (DirectoryLoader)
- [x] Create unified document pipeline

### Phase 3: Text Processing

- [x] Implement RecursiveCharacterTextSplitter
- [x] Configure chunk size and overlap
- [x] Add preprocessing utilities

### Phase 4: Embeddings

- [x] Set up HuggingFace MiniLM-L6-v2
- [x] Create embedding pipeline
- [x] Test embedding generation

### Phase 5: Vector Storage

- [x] Configure Pinecone connection
- [x] Implement vector indexing
- [x] Create upsert/query functions

### Phase 6: Retrieval & Generation

- [x] Implement RetrievalQA chain
- [x] Set up FLAN-T5-Base LLM
- [x] Create query interface

### Phase 7: CLI & Integration

- [x] Build command-line interface
- [x] Create sample documents
- [x] Integration testing

### Phase 8: CI/CD & Containerization

- [x] Create Dockerfile
- [x] Create docker-compose.yml
- [x] Set up GitHub Actions CI/CD
- [x] Add release workflow

## Configuration

| Setting             | Value                                  |
| ------------------- | -------------------------------------- |
| Embedding Model     | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding Dimension | 384                                    |
| LLM Model           | google/flan-t5-base                    |
| Chunk Size          | 1000                                   |
| Chunk Overlap       | 200                                    |
| Top K Results       | 4                                      |

## Tech Stack

- **Orchestration**: LangChain
- **Embeddings**: HuggingFace Transformers
- **Vector Store**: Pinecone
- **LLM**: FLAN-T5-Base
- **Container**: Docker

## Author

Tha Vu
