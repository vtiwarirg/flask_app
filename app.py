"""Main Flask application entry point"""
import logging
import os
import warnings

# Suppress transformers warning (we use OpenAI embeddings, not local models)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

from flask import Flask, render_template, request, flash, redirect, url_for
from config import Config
from document_processor.document_processor import DocumentProcessor
from services import RAGService, DocumentService
from routes import pages_bp, documents_bp, conversation_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    valid, message = config_class.validate()
    if not valid:
        logger.warning(f"Configuration warning: {message}")
    
    # Initialize extensions dictionary
    app.extensions = {}
    
    # Initialize application components
    initialize_app(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app


def initialize_app(app):
    # Create necessary folders
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)
    logger.info("Application folders created")
    
    # Initialize document processor
    try:
        doc_processor = DocumentProcessor(
            persist_directory=Config.CHROMA_DB_PATH,
            openai_api_key=Config.OPENAI_API_KEY
        )
        logger.info("Document processor initialized successfully")
        
        # Initialize services
        doc_service = DocumentService(doc_processor)
        rag_service = RAGService(doc_processor, Config.OPENAI_API_KEY)
        
        # Store in app extensions
        app.extensions['doc_processor'] = doc_processor
        app.extensions['doc_service'] = doc_service
        app.extensions['rag_service'] = rag_service
        
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {str(e)}")
        app.extensions['doc_processor'] = None
        app.extensions['doc_service'] = None
        app.extensions['rag_service'] = None


def register_blueprints(app):
    """Register Flask blueprints
    
    Args:
        app: Flask application instance
    """
    app.register_blueprint(pages_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(conversation_bp)
    logger.info("Blueprints registered")


def register_error_handlers(app):
    """Register error handlers
    
    Args:
        app: Flask application instance
    """
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
        return redirect(url_for('pages.documents_page'))
    
    logger.info("Error handlers registered")


# Create application instance
app = create_app()


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("Starting Flask Application")
    logger.info("=" * 70)
    logger.info(f"Document processor: {'Available' if app.extensions.get('doc_processor') else 'Not available'}")
    logger.info(f"OpenAI API: {'Configured' if Config.OPENAI_API_KEY else 'Not configured'}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Server: http://{Config.HOST}:{Config.PORT}")
    logger.info("=" * 70)
    
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
