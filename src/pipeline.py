"""
Local RAG System - RAG Pipeline

Main orchestration module combining all components into a unified RAG pipeline.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from .config import config, get_config
from .document_loader import DocumentLoader, load_documents
from .text_processor import TextProcessor, create_chunks
from .embeddings import EmbeddingManager, get_embeddings
from .vector_store import VectorStoreManager, upsert_documents
from .retrieval import RetrievalManager, RetrievalResult
from .generator import LLMGenerator, get_llm

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics about pipeline execution."""
    documents_loaded: int
    chunks_created: int
    vectors_indexed: int
    
    def __str__(self) -> str:
        return (
            f"Loaded: {self.documents_loaded} docs, "
            f"Chunked: {self.chunks_created}, "
            f"Indexed: {self.vectors_indexed}"
        )


class RAGPipeline:
    """
    Unified RAG Pipeline orchestrating all components.
    
    Provides a high-level interface for:
    - Document ingestion and indexing
    - Query processing and answer generation
    
    Example:
        pipeline = RAGPipeline()
        
        # Ingest documents
        stats = pipeline.ingest("./data/documents")
        
        # Query
        result = pipeline.query("What is the main topic?")
        print(result.answer)
    """
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store_manager = VectorStoreManager()
        self.llm_generator = LLMGenerator()
        
        self._retrieval_manager: Optional[RetrievalManager] = None
    
    @property
    def retrieval_manager(self) -> RetrievalManager:
        """Get the retrieval manager (lazy initialization)."""
        if self._retrieval_manager is None:
            llm = self.llm_generator.get_langchain_llm()
            self._retrieval_manager = RetrievalManager(
                llm=llm,
                vector_store_manager=self.vector_store_manager,
            )
        return self._retrieval_manager
    
    def ingest(
        self,
        source: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> PipelineStats:
        """
        Ingest documents from a file or directory.
        
        Args:
            source: Path to file or directory
            chunk_size: Override chunk size
            chunk_overlap: Override chunk overlap
            
        Returns:
            PipelineStats with ingestion statistics
        """
        source_path = Path(source)
        
        self.logger.info(f"Starting ingestion from: {source}")
        
        # Step 1: Load documents
        documents = load_documents(source_path)
        self.logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            return PipelineStats(
                documents_loaded=0,
                chunks_created=0,
                vectors_indexed=0,
            )
        
        # Step 2: Create chunks
        if chunk_size or chunk_overlap:
            processor = TextProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = processor.split_documents(documents)
        else:
            chunks = self.text_processor.split_documents(documents)
        
        self.logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Index to vector store
        ids = self.vector_store_manager.upsert_documents(chunks)
        self.logger.info(f"Indexed {len(ids)} vectors")
        
        stats = PipelineStats(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            vectors_indexed=len(ids),
        )
        
        self.logger.info(f"Ingestion complete: {stats}")
        
        return stats
    
    def query(self, question: str) -> RetrievalResult:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask
            
        Returns:
            RetrievalResult with answer and sources
        """
        self.logger.info(f"Processing query: {question[:50]}...")
        
        result = self.retrieval_manager.query(question)
        
        self.logger.info(
            f"Generated answer with {result.retrieval_count} sources"
        )
        
        return result
    
    def batch_query(self, questions: List[str]) -> List[RetrievalResult]:
        """
        Process multiple queries.
        
        Args:
            questions: List of questions
            
        Returns:
            List of RetrievalResult objects
        """
        return self.retrieval_manager.batch_query(questions)
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents without generation.
        
        Args:
            query: Search query
            k: Number of documents
            
        Returns:
            List of relevant documents
        """
        return self.retrieval_manager.retrieve_documents(query, k=k)
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "vector_store": self.vector_store_manager.get_index_stats(),
            "embedding_model": self.embedding_manager.get_model_info(),
            "llm_model": self.llm_generator.get_model_info(),
            "config": {
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
                "top_k": config.retrieval.top_k,
            },
        }
    
    def reset(self) -> None:
        """Reset the vector store (delete all vectors)."""
        self.vector_store_manager.delete_all()
        self.logger.info("Vector store reset complete")


def create_pipeline() -> RAGPipeline:
    """Create and return a configured RAG pipeline."""
    return RAGPipeline()
