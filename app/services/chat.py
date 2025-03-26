
"""
Chat completion service for generating RAG responses through OpenRouter.
"""

import logging
from typing import List, Dict, Any
import openai
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatService:
    """Service for chat completions through OpenRouter's OpenAI-compatible API."""
    
    def __init__(self):
        """Initialize chat clients based on configuration."""
        self.client = openai.OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )
        self.model = settings.llm_model
        logger.info(f"Initialized chat service with OpenRouter ({self.model})")
    
    async def generate_answer(self, query: str, context_blocks: List[Dict[str, Any]]) -> str:
        """
        Generate RAG answer using context blocks.
        
        Args:
            query: User's question
            context_blocks: Retrieved chunks with metadata
            
        Returns:
            Generated answer with citations
        """
        # Build context string with citations
        context_parts = []
        for block in context_blocks:
            chunk_id = block.get('chunk_id', 'unknown')
            text = block.get('text', '')
            context_parts.append(f"[{chunk_id}] {text}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a helpful AI assistant for customer support that answers questions based on provided context.

                            IMPORTANT RULES:
                            1. For questions about policies, returns, shipping, sizing, or support: Answer ONLY using the provided context and include citations
                            2. For general greetings or casual conversation: You can respond naturally and friendly
                            3. For questions outside your knowledge base: Politely redirect to relevant policies or suggest contacting support
                            4. Always include citations [chunk_id] when using context information
                            5. Be concise but comprehensive
                            6. Maintain a helpful, professional tone"""

        user_prompt = f"""Context:
                        {context}

                        Question: {query}

                        Please provide an answer based on the context above, including appropriate citations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=settings.temperature,
                max_tokens=1000
            )
            answer = response.choices[0].message.content

            logger.info("Generated answer using OpenRouter")
            return answer or "I couldn't generate an answer."
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"I encountered an error while processing your question: {str(e)}"


# Global service instance
chat_service = ChatService()
