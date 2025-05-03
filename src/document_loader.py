"""
Local RAG System - Document Loader Module

Provides document loading capabilities for PDF and TXT files using LangChain.
Implements DirectoryLoader and PyPDFLoader for flexible document ingestion.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document

from .config import config

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Exception raised when document loading fails."""
    pass


class DocumentLoader:
    """
    Unified document loader supporting multiple file formats.
    
    Supports:
    - PDF files via PyPDFLoader
    - TXT files via TextLoader
    - Directory loading with glob patterns
    
    Example:
        loader = DocumentLoader()
        docs = loader.load_directory("./data/documents")
        # or
        docs = loader.load_file("./data/document.pdf")
    """
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
    }
    
    def __init__(self):
        """Initialize the document loader."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a single file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            List of Document objects
            
        Raises:
            DocumentLoadError: If file cannot be loaded
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported file type: {extension}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        try:
            loader_class = self.SUPPORTED_EXTENSIONS[extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            self.logger.info(
                f"Loaded {len(documents)} document(s) from {file_path.name}"
            )
            
            return documents
            
        except Exception as e:
            raise DocumentLoadError(
                f"Failed to load {file_path}: {str(e)}"
            ) from e
    
    def load_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects (one per page)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if file_path.suffix.lower() != ".pdf":
            raise DocumentLoadError(f"Not a PDF file: {file_path}")
        
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["file_name"] = file_path.name
            
            self.logger.info(
                f"Loaded PDF: {file_path.name} ({len(documents)} pages)"
            )
            
            return documents
            
        except Exception as e:
            raise DocumentLoadError(
                f"Failed to load PDF {file_path}: {str(e)}"
            ) from e
    
    def load_text(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List containing a single Document
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source_type"] = "text"
                doc.metadata["file_name"] = file_path.name
            
            self.logger.info(f"Loaded text file: {file_path.name}")
            
            return documents
            
        except Exception as e:
            raise DocumentLoadError(
                f"Failed to load text file {file_path}: {str(e)}"
            ) from e
    
    def load_directory(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "**/*",
        recursive: bool = True,
        show_progress: bool = True,
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file matching
            recursive: Whether to search subdirectories
            show_progress: Whether to show loading progress
            
        Returns:
            List of all loaded Document objects
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise DocumentLoadError(f"Not a directory: {directory_path}")
        
        all_documents: List[Document] = []
        loaded_files = 0
        failed_files = 0
        
        # Find all supported files
        pattern = glob_pattern if recursive else glob_pattern.replace("**", "*")
        
        for extension in self.SUPPORTED_EXTENSIONS.keys():
            file_pattern = f"{pattern}{extension}"
            files = list(directory_path.glob(file_pattern))
            
            for file_path in files:
                try:
                    docs = self.load_file(file_path)
                    all_documents.extend(docs)
                    loaded_files += 1
                except DocumentLoadError as e:
                    self.logger.warning(f"Skipping {file_path}: {e}")
                    failed_files += 1
        
        self.logger.info(
            f"Directory loading complete: {loaded_files} files loaded, "
            f"{failed_files} failed, {len(all_documents)} total documents"
        )
        
        return all_documents
    
    def load_from_config(self) -> List[Document]:
        """
        Load documents from the configured data directory.
        
        Returns:
            List of all loaded Document objects
        """
        return self.load_directory(config.data_dir)


def load_documents(path: Union[str, Path]) -> List[Document]:
    """
    Convenience function to load documents from a file or directory.
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of Document objects
    """
    loader = DocumentLoader()
    path = Path(path)
    
    if path.is_dir():
        return loader.load_directory(path)
    else:
        return loader.load_file(path)


def get_document_stats(documents: List[Document]) -> dict:
    """
    Get statistics about loaded documents.
    
    Args:
        documents: List of documents to analyze
        
    Returns:
        Dictionary with document statistics
    """
    total_chars = sum(len(doc.page_content) for doc in documents)
    sources = set(doc.metadata.get("source", "unknown") for doc in documents)
    
    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "average_length": total_chars // len(documents) if documents else 0,
        "unique_sources": len(sources),
        "sources": list(sources),
    }
