"""Retrieves and formats query contexts for LLM generation."""

from typing import Tuple, List, Dict
from loguru import logger

from src.database.vector_store import VectorStore


class Retriever:
    """Manages context retrieval.
    
    Connects raw user queries to the vector database, formats the resulting
    chunks, and prepares citations for generation.
    """

    def __init__(self, vector_store: VectorStore):
        """Initialize Retriever with a VectorStore instance.
        
        Args:
            vector_store: Instantiated vector store connected to the embedding engine.
        """
        self.vector_store = vector_store

    def retrieve_context(self, query: str, initial_k: int = 50, final_k: int = 4) -> Tuple[str, List[Dict]]:
        """Retrieve and format best-matching contexts for a given query.
        
        Args:
            query: The user question.
            initial_k: Broad number of dense candidates to pull down.
            final_k: The final number of reranked candidates to return.
            
        Returns:
            Tuple of:
             - A single formatted string containing all text context.
             - The raw list of dictionaries from the search (for citations).
        """
        logger.info(f"Retrieving context for query: '{query}'")
        results = self.vector_store.search(query, initial_k=initial_k, final_k=final_k)
        
        if not results:
            logger.warning("No context found for the query.")
            return "", []
            
        context_blocks = []
        for i, hit in enumerate(results):
            meta = hit.get("metadata", {})
            source = meta.get("source_pdf", "Unknown Source")
            page = meta.get("page_number", "Unknown Page")
            text = hit.get("text", "").strip()
            
            # Format explicitly for LLM recognition
            block = f"--- Document: {source} (Page: {page}) ---\n{text}"
            context_blocks.append(block)
            
        # Join into mega-string with delineations
        formatted_context = "\n\n".join(context_blocks)
        
        logger.info(f"Assembled {len(context_blocks)} chunks into context block.")
        return formatted_context, results
