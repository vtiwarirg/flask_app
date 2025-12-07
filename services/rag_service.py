"""RAG (Retrieval Augmented Generation) service"""
import logging
from typing import Dict, Any, Optional, List
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)


class RAGService:
    
    def __init__(self, doc_processor, openai_api_key: str):
        self.doc_processor = doc_processor
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
    
    def search_documents(self, query: str) -> tuple[List[Dict], str]:
        """Search documents with improved query handling"""
        sources = []
        context = ""
        
        try:
            # Perform primary search
            search_results = self.doc_processor.search_documents(
                query,
                n_results=Config.SEARCH_RESULTS_LIMIT
            )
            
            # For person-related queries, also search for individual name parts
            # This helps when "Umesh Chandra" might be in different formats
            if not search_results and (' ' in query):
                logger.info("Trying expanded search with query parts...")
                query_parts = query.split()
                for part in query_parts:
                    if len(part) > 3:  # Only search meaningful words
                        extra_results = self.doc_processor.search_documents(
                            part,
                            n_results=5
                        )
                        search_results.extend(extra_results)
                
                # Remove duplicates based on chunk index
                seen = set()
                unique_results = []
                for r in search_results:
                    chunk_id = r.get('metadata', {}).get('chunk_index')
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        unique_results.append(r)
                search_results = unique_results[:Config.SEARCH_RESULTS_LIMIT]
            
            if search_results:
                context_parts = []
                
                # Filter by minimum similarity threshold
                filtered_results = [
                    r for r in search_results 
                    if r.get('similarity_score', 0) >= Config.MIN_SIMILARITY_THRESHOLD
                ]
                
                if not filtered_results:
                    logger.info(f"No results above similarity threshold {Config.MIN_SIMILARITY_THRESHOLD}")
                    return sources, context
                
                logger.info(f"Using {len(filtered_results)} results above threshold")
                
                for idx, result in enumerate(filtered_results, 1):
                    metadata = result.get('metadata', {})
                    content = result.get('content', '')
                    similarity = result.get('similarity_score', 0)
                    
                    # Add to sources
                    sources.append({
                        'filename': metadata.get('filename', 'Unknown'),
                        'content': content,
                        'content_preview': content[:200] + '...' if len(content) > 200 else content,
                        'chunk_index': metadata.get('chunk_index', 0),
                        'total_chunks': metadata.get('total_chunks', 0),
                        'similarity': similarity
                    })
                    
                    # Build context with relevance indicators
                    filename = metadata.get('filename', 'Unknown')
                    chunk_info = f"(Chunk {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)})"
                    relevance = f"[Relevance: {similarity * 100:.1f}%]" if similarity else ""
                    
                    context_parts.append(
                        f"--- Source {idx}: {filename} {chunk_info} {relevance} ---\n"
                        f"{content}\n"
                    )
                
                context = "\n\n".join(context_parts)
                logger.info(f"RAG: Built context from {len(sources)} sources (total {len(context)} chars)")
                
                # Log top result for debugging
                if sources:
                    top_source = sources[0]
                    logger.info(
                        f"Top result: {top_source['filename']} "
                        f"(similarity: {top_source['similarity']*100:.1f}%) - "
                        f"{top_source['content'][:100]}..."
                    )
        except Exception as e:
            logger.error(f"RAG search error: {str(e)}", exc_info=True)
        
        return sources, context
    
    def build_messages(self, user_message: str, context: str, history: Optional[List] = None) -> List[Dict]:
        """
        Build conversation messages with improved context handling
        """
        system_prompt = (
            "You are an intelligent assistant with access to a knowledge base. "
            "Your primary task is to answer questions using the provided document context.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. When document context is provided, use it as your primary source of truth\n"
            "2. Extract and synthesize information from the context to answer the question\n"
            "3. If the context contains names, roles, or details mentioned in the question, use them in your answer\n"
            "4. Do NOT say 'the document does not contain information' if relevant details ARE present in the context\n"
            "5. Be specific and cite what you found in the documents\n"
            "6. If the exact answer isn't in the context, share what IS available and then supplement with general knowledge\n"
            "7. Always aim to be helpful and provide value to the user"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if history:
            for item in history[-Config.HISTORY_LIMIT:]:
                user_msg = item.get('user', '')
                ai_msg = item.get('ai', '')
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if ai_msg:
                    messages.append({"role": "assistant", "content": ai_msg})
        
        # Add current question with context
        if context:
            user_content = (
                f"DOCUMENT CONTEXT (USE THIS TO ANSWER):\n"
                f"{'='*60}\n"
                f"{context}\n"
                f"{'='*60}\n\n"
                f"USER QUESTION:\n"
                f"{user_message}\n\n"
                f"TASK: Using the document context above, provide a detailed and specific answer. "
                f"Look carefully at the information provided and extract relevant details to answer the question."
            )
        else:
            user_content = (
                f"USER QUESTION: {user_message}\n\n"
                f"Note: No document context was found for this question. "
                f"Please provide a helpful general answer based on your knowledge."
            )
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def generate_response(
        self,
        user_message: str,
        use_rag: bool = True,
        history: Optional[List] = None
    ) -> Dict[str, Any]:
        sources = []
        context = ""
        
        # Search documents if RAG is enabled
        if use_rag and self.doc_processor:
            sources, context = self.search_documents(user_message)
        
        # Build messages
        messages = self.build_messages(user_message, context, history)
        
        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                temperature=Config.OPENAI_TEMPERATURE,
                max_tokens=Config.OPENAI_MAX_TOKENS
            )
            
            ai_response = response.choices[0].message.content
            
            logger.info(
                f"OpenAI response generated: {len(ai_response)} chars, "
                f"{response.usage.total_tokens} tokens, "
                f"{len(sources)} sources used"
            )
            
            if sources:
                avg_similarity = sum(s.get('similarity', 0) for s in sources) / len(sources)
                logger.info(f"Average source relevance: {avg_similarity * 100:.1f}%")
            
            return {
                'success': True,
                'response': ai_response,
                'sources': sources,
                'used_rag': use_rag and len(sources) > 0,
                'tokens_used': response.usage.total_tokens,
                'context_length': len(context) if context else 0
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f'AI service error: {str(e)}'
            }
