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
- [ ] Create requirements.txt
- [ ] Create configuration module

### Phase 2: Document Ingestion

- [ ] Implement PDF loader (PyPDFLoader)
- [ ] Implement TXT loader (DirectoryLoader)
- [ ] Create unified document pipeline

### Phase 3: Text Processing

- [ ] Implement RecursiveCharacterTextSplitter
- [ ] Configure chunk size and overlap
- [ ] Add preprocessing utilities

### Phase 4: Embeddings

- [ ] Set up HuggingFace MiniLM-L6-v2
- [ ] Create embedding pipeline
- [ ] Test embedding generation

### Phase 5: Vector Storage

- [ ] Configure Pinecone connection
- [ ] Implement vector indexing
- [ ] Create upsert/query functions

### Phase 6: Retrieval & Generation

- [ ] Implement RetrievalQA chain
- [ ] Set up FLAN-T5-Base LLM
- [ ] Create query interface

### Phase 7: CLI & Integration

- [ ] Build command-line interface
- [ ] Create sample documents
- [ ] Integration testing

### Phase 8: CI/CD & Containerization

- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Set up GitHub Actions CI/CD
- [ ] Add release workflow

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
