from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import os
import logging
from werkzeug.utils import secure_filename
from document_processor.document_processor import DocumentProcessor
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['CHROMA_DB_PATH'] = os.path.join(os.path.dirname(__file__), 'chroma_db')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHROMA_DB_PATH'], exist_ok=True)

# Validate OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
    logger.warning("OpenAI API key not configured properly in .env file!")

# Initialize document processor
try:
    doc_processor = DocumentProcessor(
        persist_directory=app.config['CHROMA_DB_PATH'],
        openai_api_key=openai_api_key
    )
    logger.info("Document processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize document processor: {str(e)}")
    doc_processor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    try:
        return render_template('index.html', site_name='Document AI Assistant',chatbot_available=doc_processor is not None,intents_count=0)
    except Exception as e:
        return render_template('error.html', error="Page not found"), 404

@app.route('/chat')
def chat():
    try:
        stats = None
        if doc_processor:
            try:
                stats = doc_processor.get_collection_stats()
            except:
                stats = {'total_documents': 0, 'total_chunks': 0}
        
        return render_template('chat.html', stats=stats, processor_available=doc_processor is not None)
    except Exception as e:
        logger.error(f"Error rendering chat page: {str(e)}")
        return render_template('error.html', error="Chat page unavailable"), 500

@app.route('/documents')
def documents():
    if not doc_processor:
        return redirect(url_for('home'))
    
    try:
        all_docs = doc_processor.get_all_documents()
        return render_template('documents.html', documents=all_docs, max_file_size_mb=app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024))
    except Exception as e:
        flash(f'Error loading documents: {str(e)}', 'error')
        return render_template('documents.html', documents=[])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('documents'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('documents'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get file extension
        file_type = filename.rsplit('.', 1)[1].lower()
        
        # Process and store document
        result = doc_processor.process_and_store_document(file_path, file_type, filename)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        if result['success']:
            flash(f'Successfully uploaded {filename}! Created {result["chunks_count"]} chunks.', 'success')
        else:
            flash(f'Error: {result["error"]}', 'error')
    else:
        flash('Invalid file type. Only .txt, .pdf, and .docx files are allowed.', 'error')
    
    return redirect(url_for('documents'))

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    results = doc_processor.search_documents(query, n_results=5)
    
    return jsonify({
        'success': True,
        'query': query,
        'results': results
    })

@app.route('/delete/<file_hash>', methods=['POST'])
def delete_document(file_hash):
    success = doc_processor.delete_document(file_hash)
    
    if success:
        flash('Document deleted successfully', 'success')
    else:
        flash('Error deleting document', 'error')
    
    return redirect(url_for('documents'))

@app.route('/api/conversation/<string:conversation_id>/messages', methods=['POST'])
def conversation(conversation_id):
    if not doc_processor:
        return jsonify({
            'success': False,
            'error': 'Document processor not available'
        }), 500
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        use_rag = data.get('use_rag', True)
        history = data.get('history', [])
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'Message is required'
            }), 400
        
        logger.info(f"Conversation {conversation_id}: '{user_message}' (RAG: {use_rag})")
        
        # Generate response
        response_data = generate_rag_response(user_message, use_rag, history)
        response_data['conversation_id'] = conversation_id
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in conversation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_rag_response(user_message, use_rag=True, history=None):
    """Generate AI response with optional RAG"""
    sources = []
    context = ""
    
    # Search documents if RAG is enabled
    if use_rag:
        try:
            search_results = doc_processor.search_documents(user_message, n_results=3)
            
            if search_results:
                # Map search results to expected format for sources
                sources = []
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
                    context_parts.append(f"From {metadata.get('filename', 'Unknown')}:\n{content}")
                
                context = "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            sources = []
            context = ""
    
    # Build conversation context
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Answer questions based on the provided context when available. Be concise and accurate."
        }
    ]
    
    # Add history if available
    if history:
        for item in history[-5:]:  # Last 5 exchanges
            messages.append({"role": "user", "content": item.get('user', '')})
            messages.append({"role": "assistant", "content": item.get('ai', '')})
    
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
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content
        
        return {
            'success': True,
            'response': ai_response,
            'sources': sources,
            'used_rag': use_rag and len(sources) > 0
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'AI service error: {str(e)}'
        }

@app.route('/api/conversation/<string:conversation_id>/history', methods=['GET'])
def get_conversation_history(conversation_id):
    # TODO: Implement proper session management with database
    logger.info(f"Retrieving history for conversation: {conversation_id}")
    return jsonify({
        'success': True,
        'conversation_id': conversation_id,
        'history': [],
        'message': 'Session management not yet implemented'
    })

@app.route('/api/conversation/<string:conversation_id>/clear', methods=['POST'])
def clear_conversation(conversation_id):
    return jsonify({
        'success': True,
        'conversation_id': conversation_id,
        'message': 'Conversation cleared'
    })


if __name__ == '__main__':
    app.run(debug=True)
    