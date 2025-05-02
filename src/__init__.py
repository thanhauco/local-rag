"""
Local RAG System

A complete Retrieval-Augmented Generation system built from scratch.
Supports document ingestion, embedding, vector storage, and LLM-powered generation.
"""

from .config import config, get_config, reload_config

__version__ = "1.0.0"
__author__ = "Tha Vu"

__all__ = [
    "config",
    "get_config", 
    "reload_config",
    "__version__",
    "__author__",
]
