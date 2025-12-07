from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import os
import logging
from werkzeug.utils import secure_filename
from document_processor.document_processor import DocumentProcessor
from openai import OpenAI
from typing import Dict, Any, Optional, Tuple

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Application configuration"""
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
    OPENAI_MODEL = 'gpt-3.5-turbo'
    OPENAI_TEMPERATURE = 0.7
    OPENAI_MAX_TOKENS = 500
    SEARCH_RESULTS_LIMIT = 3
    HISTORY_LIMIT = 5

# Apply configuration
app.config.from_object(Config)

# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_app():
    """Initialize application components"""
    # Create necessary folders
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
    logger.info("Application folders created")
    
    # Validate OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
        logger.warning("OpenAI API key not configured properly in .env file!")
        return None, openai_api_key
    
    # Initialize document processor
    try:
        doc_processor = DocumentProcessor(
            persist_directory=Config.CHROMA_DB_PATH,
            openai_api_key=openai_api_key
        )
        logger.info("Document processor initialized successfully")
        return doc_processor, openai_api_key
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {str(e)}")
        return None, openai_api_key

# Initialize components
doc_processor, openai_api_key = initialize_app()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def create_error_response(message: str, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
    """Create standardized error response"""
    return jsonify({
        'success': False,
        'error': message
    }), status_code

def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized success response"""
    response = {'success': True}
    response.update(data)
    return jsonify(response)

# ============================================================================
# ROUTES - PAGES
# ============================================================================
@app.route('/')
def home():
    """Render home page"""
    try:
        return render_template(
            'index.html',
            site_name='Document AI Assistant',
            chatbot_available=doc_processor is not None,
            intents_count=0
        )
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return render_template('error.html', error="Page not found"), 404

@app.route('/chat')
def chat():
    """Render chat interface page"""
    try:
        stats = {'total_documents': 0, 'total_chunks': 0}
        if doc_processor:
            try:
                stats = doc_processor.get_collection_stats()
            except Exception as e:
                logger.warning(f"Failed to get collection stats: {str(e)}")
        
        return render_template(
            'chat.html',
            stats=stats,
            processor_available=doc_processor is not None
        )
    except Exception as e:
        logger.error(f"Error rendering chat page: {str(e)}")
        return render_template('error.html', error="Chat page unavailable"), 500

@app.route('/documents')
def documents():
    """Render document management page"""
    if not doc_processor:
        flash('Document processor not available', 'warning')
        return redirect(url_for('home'))
    
    try:
        all_docs = doc_processor.get_all_documents()
        max_file_size_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        
        return render_template(
            'documents.html',
            documents=all_docs,
            max_file_size_mb=max_file_size_mb
        )
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        flash(f'Error loading documents: {str(e)}', 'error')
        return render_template('documents.html', documents=[])

# ============================================================================
# ROUTES - DOCUMENT OPERATIONS
# ============================================================================
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    # Validate file presence
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('documents'))
    
    file = request.files['file']
    
    if not file.filename:
        flash('No file selected', 'error')
        return redirect(url_for('documents'))
    
    # Validate file type
    if not allowed_file(file.filename):
        allowed = ', '.join(Config.ALLOWED_EXTENSIONS)
        flash(f'Invalid file type. Allowed: {allowed}', 'error')
        return redirect(url_for('documents'))
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        # Save file temporarily
        file.save(file_path)
        logger.info(f"File saved: {filename}")
        
        # Get file extension
        file_type = filename.rsplit('.', 1)[1].lower()
        
        # Process and store document
        result = doc_processor.process_and_store_document(
            file_path, file_type, filename
        )
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file removed: {filename}")
        
        # Handle result
        if result['success']:
            chunks = result.get('chunks_count', 0)
            flash(f'Successfully uploaded {filename}! Created {chunks} chunks.', 'success')
        else:
            error_msg = result.get('error', 'Unknown error')
            flash(f'Error processing file: {error_msg}', 'error')
            
    except Exception as e:
        logger.error(f"Upload error for {filename}: {str(e)}")
        flash(f'Upload failed: {str(e)}', 'error')
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
    
    return redirect(url_for('documents'))

@app.route('/search', methods=['POST'])
def search():
    """Search documents using semantic search"""
    if not doc_processor:
        return create_error_response('Document processor not available', 500)
    
    query = request.form.get('query', '').strip()
    
    if not query:
        return create_error_response('Query is required')
    
    try:
        results = doc_processor.search_documents(query, n_results=5)
        logger.info(f"Search query: '{query}' - Found {len(results)} results")
        
        return create_success_response({
            'query': query,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return create_error_response(f'Search failed: {str(e)}', 500)

@app.route('/delete/<file_hash>', methods=['POST'])
def delete_document(file_hash: str):
    """Delete a document by file hash"""
    if not doc_processor:
        flash('Document processor not available', 'error')
        return redirect(url_for('documents'))
    
    if not file_hash:
        flash('Invalid document identifier', 'error')
        return redirect(url_for('documents'))
    
    try:
        success = doc_processor.delete_document(file_hash)
        
        if success:
            logger.info(f"Document deleted: {file_hash}")
            flash('Document deleted successfully', 'success')
        else:
            logger.warning(f"Failed to delete document: {file_hash}")
            flash('Error deleting document', 'error')
    except Exception as e:
        logger.error(f"Delete error for {file_hash}: {str(e)}")
        flash(f'Delete failed: {str(e)}', 'error')
    
    return redirect(url_for('documents'))

# ============================================================================
# ROUTES - CONVERSATION API
# ============================================================================
@app.route('/api/conversation/<string:conversation_id>/messages', methods=['POST'])
def conversation(conversation_id: str):
    """Handle conversation messages with RAG support"""
    if not doc_processor:
        return create_error_response('Document processor not available', 500)
    
    if not openai_api_key:
        return create_error_response('OpenAI API key not configured', 500)
    
    try:
        # Parse and validate request
        data = request.get_json()
        if not data:
            return create_error_response('Invalid JSON payload')
        
        user_message = data.get('message', '').strip()
        use_rag = data.get('use_rag', True)
        history = data.get('history', [])
        
        if not user_message:
            return create_error_response('Message is required')
        
        logger.info(f"Conversation {conversation_id}: '{user_message[:50]}...' (RAG: {use_rag})")
        
        # Generate response
        response_data = generate_rag_response(user_message, use_rag, history)
        response_data['conversation_id'] = conversation_id
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Conversation error: {str(e)}", exc_info=True)
        return create_error_response(f'Conversation failed: {str(e)}', 500)

def generate_rag_response(user_message: str, use_rag: bool = True, history: Optional[list] = None) -> Dict[str, Any]:
    """Generate AI response with optional RAG
    
    Args:
        user_message: User's input message
        use_rag: Whether to use RAG (Retrieval Augmented Generation)
        history: Conversation history
    
    Returns:
        Dict containing success status, response, sources, and RAG usage info
    """
    sources = []
    context = ""
    
    # Search documents if RAG is enabled
    if use_rag and doc_processor:
        try:
            search_results = doc_processor.search_documents(
                user_message,
                n_results=Config.SEARCH_RESULTS_LIMIT
            )
            
            if search_results:
                context_parts = []
                
                for result in search_results:
                    metadata = result.get('metadata', {})
                    content = result.get('content', '')
                    
                    # Add to sources with expected structure
                    sources.append({
                        'filename': metadata.get('filename', 'Unknown'),
                        'content': content,
                        'content_preview': content[:200] + '...' if len(content) > 200 else content,
                        'chunk_index': metadata.get('chunk_index', 0),
                        'similarity': result.get('similarity_score', 0)
                    })
                    
                    # Build context for AI
                    filename = metadata.get('filename', 'Unknown')
                    context_parts.append(f"From {filename}:\n{content}")
                
                context = "\n\n".join(context_parts)
                logger.info(f"RAG: Found {len(sources)} relevant documents")
        except Exception as e:
            logger.error(f"RAG search error: {str(e)}", exc_info=True)
            sources = []
            context = ""
    
    # Build conversation context
    system_prompt = (
        "You are a helpful AI assistant. Answer questions based on the provided "
        "context when available. Be concise, accurate, and cite sources when using "
        "document context."
    )
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
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
        user_content = f"Context from documents:\n{context}\n\nQuestion: {user_message}"
    else:
        user_content = user_message
    
    messages.append({"role": "user", "content": user_content})
    
    # Call OpenAI API
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            temperature=Config.OPENAI_TEMPERATURE,
            max_tokens=Config.OPENAI_MAX_TOKENS
        )
        
        ai_response = response.choices[0].message.content
        
        logger.info(f"OpenAI response generated (tokens: {response.usage.total_tokens})")
        
        return {
            'success': True,
            'response': ai_response,
            'sources': sources,
            'used_rag': use_rag and len(sources) > 0,
            'tokens_used': response.usage.total_tokens
        }
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': f'AI service error: {str(e)}'
        }

@app.route('/api/conversation/<string:conversation_id>/history', methods=['GET'])
def get_conversation_history(conversation_id: str):
    """Get conversation history (placeholder for future implementation)"""
    logger.info(f"History request for conversation: {conversation_id}")
    # TODO: Implement proper session management with database
    return create_success_response({
        'conversation_id': conversation_id,
        'history': [],
        'message': 'Session management not yet implemented'
    })

@app.route('/api/conversation/<string:conversation_id>/clear', methods=['POST'])
def clear_conversation(conversation_id: str):
    """Clear conversation history (placeholder for future implementation)"""
    logger.info(f"Clear request for conversation: {conversation_id}")
    # TODO: Implement session clearing
    return create_success_response({
        'conversation_id': conversation_id,
        'message': 'Conversation cleared'
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 error: {str(error)}", exc_info=True)
    return render_template('error.html', error="Internal server error"), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    max_size = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    flash(f'File too large. Maximum size: {max_size}MB', 'error')
    return redirect(url_for('documents'))

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Document processor: {'Available' if doc_processor else 'Not available'}")
    logger.info(f"OpenAI API: {'Configured' if openai_api_key else 'Not configured'}")
    app.run(debug=True, host='127.0.0.1', port=5000)
