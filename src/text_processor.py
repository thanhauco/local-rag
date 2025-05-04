"""
Local RAG System - Text Processor Module

Provides text chunking and preprocessing capabilities using LangChain's
RecursiveCharacterTextSplitter for optimal chunk generation.
"""

import logging
import re
from typing import List, Optional, Callable
from dataclasses import dataclass

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetrics:
    """Metrics about generated chunks."""
    total_chunks: int
    total_characters: int
    average_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    
    def __str__(self) -> str:
        return (
            f"Chunks: {self.total_chunks}, "
            f"Avg size: {self.average_chunk_size:.0f}, "
            f"Range: [{self.min_chunk_size}, {self.max_chunk_size}]"
        )


class TextProcessingError(Exception):
    """Exception raised when text processing fails."""
    pass


class TextProcessor:
    """
    Text processor for chunking documents using RecursiveCharacterTextSplitter.
    
    The recursive splitter tries to split on semantic boundaries (paragraphs,
    sentences, words) to maintain context within chunks.
    
    Example:
        processor = TextProcessor(chunk_size=1000, chunk_overlap=200)
        chunks = processor.split_documents(documents)
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
    ):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Maximum size of each chunk (default from config)
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separators to try (in order)
            length_function: Function to measure text length
        """
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.separators = separators or config.chunking.separators
        self.length_function = length_function
        
        self._validate_config()
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=self.length_function,
            is_separator_regex=False,
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.chunk_overlap >= self.chunk_size:
            raise TextProcessingError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        if self.chunk_size < 100:
            self.logger.warning(
                f"chunk_size ({self.chunk_size}) is very small, "
                "this may result in loss of context"
            )
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = self.splitter.split_text(text)
        
        self.logger.debug(f"Split text into {len(chunks)} chunks")
        
        return chunks
    
    def split_documents(
        self,
        documents: List[Document],
        add_metadata: bool = True,
    ) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.
        
        Args:
            documents: List of documents to split
            add_metadata: Whether to add chunk metadata
            
        Returns:
            List of chunked documents
        """
        if not documents:
            return []
        
        chunks = self.splitter.split_documents(documents)
        
        if add_metadata:
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
                chunk.metadata["total_chunks"] = len(chunks)
        
        self.logger.info(
            f"Split {len(documents)} documents into {len(chunks)} chunks"
        )
        
        return chunks
    
    def get_chunk_metrics(self, chunks: List[Document]) -> ChunkMetrics:
        """
        Calculate metrics for generated chunks.
        
        Args:
            chunks: List of chunk documents
            
        Returns:
            ChunkMetrics object with statistics
        """
        if not chunks:
            return ChunkMetrics(
                total_chunks=0,
                total_characters=0,
                average_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
            )
        
        sizes = [len(chunk.page_content) for chunk in chunks]
        
        return ChunkMetrics(
            total_chunks=len(chunks),
            total_characters=sum(sizes),
            average_chunk_size=sum(sizes) / len(sizes),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
        )


class TextPreprocessor:
    """
    Text preprocessor for cleaning and normalizing text before chunking.
    """
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Remove excessive whitespace while preserving structure."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
        """Remove special characters from text."""
        if keep_punctuation:
            # Keep alphanumeric, spaces, and common punctuation
            pattern = r'[^a-zA-Z0-9\s.,!?;:\'\"\-\(\)\[\]]'
        else:
            pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        return unicodedata.normalize('NFKD', text)
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\S+@\S+\.\S+'
        return re.sub(email_pattern, '', text)
    
    @classmethod
    def preprocess(
        cls,
        text: str,
        clean_whitespace: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
        normalize: bool = True,
    ) -> str:
        """
        Apply preprocessing pipeline to text.
        
        Args:
            text: Text to preprocess
            clean_whitespace: Whether to clean excessive whitespace
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize: Whether to normalize unicode
            
        Returns:
            Preprocessed text
        """
        if normalize:
            text = cls.normalize_unicode(text)
        if remove_urls:
            text = cls.remove_urls(text)
        if remove_emails:
            text = cls.remove_emails(text)
        if clean_whitespace:
            text = cls.clean_whitespace(text)
        
        return text


def create_chunks(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Convenience function to create chunks from documents.
    
    Args:
        documents: Documents to chunk
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents
    """
    processor = TextProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return processor.split_documents(documents)


def preprocess_documents(
    documents: List[Document],
    **kwargs,
) -> List[Document]:
    """
    Preprocess documents before chunking.
    
    Args:
        documents: Documents to preprocess
        **kwargs: Arguments passed to TextPreprocessor.preprocess
        
    Returns:
        Preprocessed documents
    """
    preprocessor = TextPreprocessor()
    
    for doc in documents:
        doc.page_content = preprocessor.preprocess(doc.page_content, **kwargs)
    
    return documents
