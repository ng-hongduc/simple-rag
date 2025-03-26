
"""
Configuration management for the RAG application.
Handles environment variables and application settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "nvidia/nemotron-3.5-lightning:free"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    environment: str = "development"
    log_level: str = "INFO"
    chunk_size: int = 400
    chunk_overlap: int = 60
    temperature: float = 0.1
    embedding_dimensions: int = 384
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
