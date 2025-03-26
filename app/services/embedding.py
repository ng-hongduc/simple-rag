
"""
Embedding service for generating text embeddings locally with BGE.
Focused solely on vector embeddings for RAG retrieval.
"""

import logging
import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Service for generating local embeddings with Sentence Transformers."""
    
    def __init__(self):
        """Load the configured local embedding model."""
        self.model = SentenceTransformer(settings.embedding_model)
        self.embed_model = settings.embedding_model
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (384 dimensions for BAAI/bge-small-en-v1.5)
        """
        try:
            embeddings = await asyncio.to_thread(
                self.model.encode,
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            embeddings = embeddings.tolist()
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([query])
        return embeddings[0]


# Global service instance
embedding_service = EmbeddingService()
