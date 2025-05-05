"""
Local RAG System - Embeddings Module

Provides embedding generation using HuggingFace's sentence-transformers.
Default model: all-MiniLM-L6-v2 (384 dimensions, fast and efficient).
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import hashlib
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for embedding results with metadata."""
    text: str
    embedding: List[float]
    model: str
    dimension: int
    
    @property
    def text_hash(self) -> str:
        """Get hash of the text."""
        return hashlib.md5(self.text.encode()).hexdigest()


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    pass


class EmbeddingManager:
    """
    Manager for generating embeddings using HuggingFace models.
    
    Uses sentence-transformers/all-MiniLM-L6-v2 by default:
    - Output dimension: 384
    - Small and fast, suitable for local execution
    - Good performance on semantic similarity tasks
    
    Example:
        manager = EmbeddingManager()
        embeddings = manager.embed_texts(["Hello, world!"])
        query_embedding = manager.embed_query("What is RAG?")
    """
    
    _instance: Optional['EmbeddingManager'] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding manager."""
        if self._embeddings is None:
            self._initialize_embeddings()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _initialize_embeddings(self) -> None:
        """Initialize the HuggingFace embeddings model."""
        try:
            self.logger.info(
                f"Loading embedding model: {config.embedding.model_name}"
            )
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=config.embedding.model_name,
                model_kwargs=config.embedding.model_kwargs,
                encode_kwargs=config.embedding.encode_kwargs,
            )
            
            # Warm up the model with a test embedding
            _ = self._embeddings.embed_query("test")
            
            self.logger.info(
                f"Embedding model loaded successfully. "
                f"Dimension: {config.embedding.dimension}"
            )
            
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize embedding model: {str(e)}"
            ) from e
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get the HuggingFace embeddings instance."""
        if self._embeddings is None:
            self._initialize_embeddings()
        return self._embeddings
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return config.embedding.model_name
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return config.embedding.dimension
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        try:
            embedding = self.embeddings.embed_query(text)
            self.logger.debug(
                f"Generated embedding for query (length: {len(embedding)})"
            )
            return embedding
            
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate query embedding: {str(e)}"
            ) from e
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if len(valid_texts) != len(texts):
            self.logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty texts"
            )
        
        try:
            embeddings = self.embeddings.embed_documents(valid_texts)
            self.logger.info(
                f"Generated {len(embeddings)} embeddings"
            )
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings: {str(e)}"
            ) from e
    
    def embed_documents(
        self,
        documents: List[Document],
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of EmbeddingResult objects
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_texts(texts)
        
        results = [
            EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model_name,
                dimension=len(embedding),
            )
            for text, embedding in zip(texts, embeddings)
        ]
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": config.embedding.device,
            "normalize": config.embedding.normalize_embeddings,
        }


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get the HuggingFace embeddings instance.
    
    Returns:
        Configured HuggingFaceEmbeddings instance
    """
    return EmbeddingManager().embeddings


def embed_query(text: str) -> List[float]:
    """
    Convenience function to embed a single query.
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector
    """
    return EmbeddingManager().embed_query(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to embed multiple texts.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    return EmbeddingManager().embed_texts(texts)


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    import math
    
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    magnitude1 = math.sqrt(sum(a * a for a in embedding1))
    magnitude2 = math.sqrt(sum(b * b for b in embedding2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)
