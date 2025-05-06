"""
Local RAG System - Vector Store Module

Provides Pinecone vector database integration for storing and retrieving
document embeddings with semantic similarity search.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from .config import config
from .embeddings import get_embeddings, EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for search results with metadata."""
    document: Document
    score: float
    id: str
    
    @property
    def content(self) -> str:
        """Get the document content."""
        return self.document.page_content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the document metadata."""
        return self.document.metadata


class VectorStoreError(Exception):
    """Exception raised when vector store operations fail."""
    pass


class VectorStoreManager:
    """
    Manager for Pinecone vector database operations.
    
    Handles:
    - Index creation and management
    - Document upsert with embeddings
    - Similarity search with scores
    - Index statistics and health checks
    
    Example:
        manager = VectorStoreManager()
        manager.upsert_documents(documents)
        results = manager.similarity_search("What is RAG?", k=4)
    """
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the vector store manager.
        
        Args:
            index_name: Pinecone index name (default from config)
            api_key: Pinecone API key (default from config)
        """
        self.index_name = index_name or config.pinecone.index_name
        self.api_key = api_key or config.pinecone.api_key
        
        self._validate_config()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client: Optional[Pinecone] = None
        self._vector_store: Optional[PineconeVectorStore] = None
        self._embeddings = get_embeddings()
    
    def _validate_config(self) -> None:
        """Validate Pinecone configuration."""
        if not self.api_key:
            raise VectorStoreError(
                "Pinecone API key is required. "
                "Set PINECONE_API_KEY in .env file."
            )
    
    @property
    def client(self) -> Pinecone:
        """Get the Pinecone client (lazy initialization)."""
        if self._client is None:
            self._client = Pinecone(api_key=self.api_key)
            self.logger.info("Pinecone client initialized")
        return self._client
    
    def index_exists(self) -> bool:
        """Check if the index exists."""
        existing_indexes = self.client.list_indexes()
        return self.index_name in [idx.name for idx in existing_indexes]
    
    def create_index(
        self,
        dimension: int = 384,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        """
        Create a new Pinecone index.
        
        Args:
            dimension: Vector dimension (384 for MiniLM-L6-v2)
            metric: Distance metric (cosine, euclidean, dotproduct)
            cloud: Cloud provider
            region: Cloud region
        """
        if self.index_exists():
            self.logger.info(f"Index '{self.index_name}' already exists")
            return
        
        try:
            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region,
                ),
            )
            
            # Wait for index to be ready
            self.logger.info(f"Creating index '{self.index_name}'...")
            while not self.client.describe_index(self.index_name).status.ready:
                time.sleep(1)
            
            self.logger.info(f"Index '{self.index_name}' created successfully")
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to create index: {str(e)}"
            ) from e
    
    def delete_index(self) -> None:
        """Delete the Pinecone index."""
        if not self.index_exists():
            self.logger.warning(f"Index '{self.index_name}' does not exist")
            return
        
        try:
            self.client.delete_index(self.index_name)
            self.logger.info(f"Index '{self.index_name}' deleted")
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete index: {str(e)}"
            ) from e
    
    def get_vector_store(self) -> PineconeVectorStore:
        """
        Get the LangChain Pinecone vector store.
        
        Returns:
            Configured PineconeVectorStore instance
        """
        if self._vector_store is None:
            if not self.index_exists():
                self.create_index(dimension=config.embedding.dimension)
            
            self._vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self._embeddings,
                pinecone_api_key=self.api_key,
            )
            
            self.logger.info("Vector store initialized")
        
        return self._vector_store
    
    def upsert_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        namespace: str = "",
    ) -> List[str]:
        """
        Upsert documents into the vector store.
        
        Args:
            documents: Documents to upsert
            batch_size: Number of documents per batch
            namespace: Pinecone namespace
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        vector_store = self.get_vector_store()
        
        # Generate unique IDs for documents
        ids = [str(uuid.uuid4()) for _ in documents]
        
        try:
            # Add documents in batches
            all_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                vector_store.add_documents(
                    documents=batch,
                    ids=batch_ids,
                    namespace=namespace,
                )
                
                all_ids.extend(batch_ids)
                self.logger.debug(
                    f"Upserted batch {i // batch_size + 1}: "
                    f"{len(batch)} documents"
                )
            
            self.logger.info(
                f"Upserted {len(all_ids)} documents to index '{self.index_name}'"
            )
            
            return all_ids
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to upsert documents: {str(e)}"
            ) from e
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            namespace: Pinecone namespace
            filter: Metadata filter
            
        Returns:
            List of SearchResult objects
        """
        vector_store = self.get_vector_store()
        
        try:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=k,
                namespace=namespace,
                filter=filter,
            )
            
            search_results = [
                SearchResult(
                    document=doc,
                    score=score,
                    id=doc.metadata.get("id", "unknown"),
                )
                for doc, score in results
            ]
            
            self.logger.info(
                f"Found {len(search_results)} results for query"
            )
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to perform similarity search: {str(e)}"
            ) from e
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        namespace: str = "",
    ) -> List[SearchResult]:
        """
        Perform similarity search using a pre-computed embedding.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            namespace: Pinecone namespace
            
        Returns:
            List of SearchResult objects
        """
        vector_store = self.get_vector_store()
        
        try:
            results = vector_store.similarity_search_by_vector_with_score(
                embedding=embedding,
                k=k,
                namespace=namespace,
            )
            
            return [
                SearchResult(
                    document=doc,
                    score=score,
                    id=doc.metadata.get("id", "unknown"),
                )
                for doc, score in results
            ]
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to perform vector search: {str(e)}"
            ) from e
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.index_exists():
            return {"error": "Index does not exist"}
        
        try:
            index = self.client.Index(self.index_name)
            stats = index.describe_index_stats()
            
            return {
                "index_name": self.index_name,
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}
    
    def delete_all(self, namespace: str = "") -> None:
        """
        Delete all vectors from the index.
        
        Args:
            namespace: Pinecone namespace (empty for all)
        """
        if not self.index_exists():
            return
        
        try:
            index = self.client.Index(self.index_name)
            index.delete(delete_all=True, namespace=namespace)
            self.logger.info(f"Deleted all vectors from '{self.index_name}'")
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete vectors: {str(e)}"
            ) from e


def get_vector_store() -> PineconeVectorStore:
    """
    Get the default vector store instance.
    
    Returns:
        Configured PineconeVectorStore
    """
    return VectorStoreManager().get_vector_store()


def upsert_documents(documents: List[Document]) -> List[str]:
    """
    Convenience function to upsert documents.
    
    Args:
        documents: Documents to upsert
        
    Returns:
        List of document IDs
    """
    return VectorStoreManager().upsert_documents(documents)


def similarity_search(query: str, k: int = 4) -> List[SearchResult]:
    """
    Convenience function for similarity search.
    
    Args:
        query: Search query
        k: Number of results
        
    Returns:
        List of SearchResult objects
    """
    return VectorStoreManager().similarity_search(query, k=k)
