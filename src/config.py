"""
Local RAG System - Configuration Module

Centralized configuration management using environment variables and dataclasses.
Provides type-safe access to all system configuration parameters.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PineconeConfig:
    """Pinecone vector database configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "local-rag-index"))
    environment: str = field(default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"))
    
    def validate(self) -> bool:
        """Validate Pinecone configuration."""
        if not self.api_key:
            raise ValueError(
                "PINECONE_API_KEY is required. "
                "Set it in .env file or as environment variable."
            )
        return True


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    dimension: int = 384  # MiniLM-L6-v2 output dimension
    device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    normalize_embeddings: bool = True
    
    @property
    def model_kwargs(self) -> dict:
        """Get model initialization kwargs."""
        return {"device": self.device}
    
    @property
    def encode_kwargs(self) -> dict:
        """Get encoding kwargs."""
        return {"normalize_embeddings": self.normalize_embeddings}


@dataclass
class LLMConfig:
    """Language model configuration."""
    model_name: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "google/flan-t5-base")
    )
    max_length: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_LENGTH", "512"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    device: str = field(default_factory=lambda: os.getenv("LLM_DEVICE", "cpu"))
    
    @property
    def pipeline_kwargs(self) -> dict:
        """Get pipeline initialization kwargs."""
        return {
            "max_length": self.max_length,
            "temperature": self.temperature,
        }


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    separators: list = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    
    def validate(self) -> bool:
        """Validate chunking configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return True


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "4"))
    )
    score_threshold: float = field(
        default_factory=lambda: float(os.getenv("SCORE_THRESHOLD", "0.5"))
    )
    search_type: str = "similarity"  # similarity, mmr


@dataclass
class Config:
    """Main configuration class aggregating all sub-configurations."""
    
    # Sub-configurations
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Paths
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "data/documents"))
    )
    
    def validate_all(self) -> bool:
        """Validate all configurations."""
        self.pinecone.validate()
        self.chunking.validate()
        logger.info("Configuration validated successfully")
        return True
    
    def __post_init__(self):
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    load_dotenv(override=True)
    config = Config()
    return config
