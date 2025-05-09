"""
Unit tests for configuration module.
"""

from src.config import Config, PineconeConfig, EmbeddingConfig

def test_config_initialization():
    """Test that config initializes with default values."""
    config = Config()
    assert config.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.chunking.chunk_size == 1000
    assert config.chunking.chunk_overlap == 200

def test_pinecone_validation():
    """Test Pinecone config validation."""
    pc_config = PineconeConfig(api_key="")
    try:
        pc_config.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "PINECONE_API_KEY is required" in str(e)

def test_chunking_validation():
    """Test chunking config validation."""
    from src.config import ChunkingConfig
    c_config = ChunkingConfig(chunk_size=500, chunk_overlap=200)
    assert c_config.validate() is True
    
    c_config_invalid = ChunkingConfig(chunk_size=200, chunk_overlap=500)
    try:
        c_config_invalid.validate()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be less than" in str(e)
