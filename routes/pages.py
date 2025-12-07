"""Page routes blueprint"""
import logging
from flask import Blueprint, render_template, redirect, url_for, flash, current_app

logger = logging.getLogger(__name__)

pages_bp = Blueprint('pages', __name__)


@pages_bp.route('/')
def home():
    """Render home page"""
    try:
        doc_processor = current_app.extensions.get('doc_processor')
        return render_template(
            'index.html',
            site_name='Document AI Assistant',
            chatbot_available=doc_processor is not None,
            intents_count=0
        )
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return render_template('error.html', error="Page not found"), 404


@pages_bp.route('/chat')
def chat():
    """Render chat interface page"""
    try:
        doc_service = current_app.extensions.get('doc_service')
        stats = {'total_documents': 0, 'total_chunks': 0}
        
        if doc_service:
            stats = doc_service.get_collection_stats()
        
        return render_template(
            'chat.html',
            site_name='Document AI Assistant',
            page_title='Chat',
            stats=stats,
            processor_available=doc_service is not None
        )
    except Exception as e:
        logger.error(f"Error rendering chat page: {str(e)}", exc_info=True)
        return render_template('error.html', error="Chat page unavailable"), 500


@pages_bp.route('/documents')
def documents_page():
    """Render document management page"""
    doc_service = current_app.extensions.get('doc_service')
    
    if not doc_service:
        flash('Document processor not available', 'warning')
        return redirect(url_for('pages.home'))
    
    try:
        all_docs = doc_service.get_all_documents()
        max_file_size_mb = current_app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
        
        return render_template(
            'documents.html',
            site_name='Document AI Assistant',
            page_title='Documents',
            documents=all_docs,
            max_file_size=max_file_size_mb
        )
    except Exception as e:
        logger.error(f"Error rendering documents page: {str(e)}", exc_info=True)
        flash('Error loading documents page', 'error')
        return redirect(url_for('pages.home'))
