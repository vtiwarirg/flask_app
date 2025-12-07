"""Document operations routes blueprint"""
import logging
from flask import Blueprint, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename
from utils import allowed_file, create_error_response, create_success_response
from config import Config

logger = logging.getLogger(__name__)

documents_bp = Blueprint('documents', __name__, url_prefix='/api/documents')


@documents_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    doc_service = current_app.extensions.get('doc_service')
    
    # Validate file presence
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('pages.documents_page'))
    
    file = request.files['file']
    
    if not file.filename:
        flash('No file selected', 'error')
        return redirect(url_for('pages.documents_page'))
    
    # Validate file type
    if not allowed_file(file.filename):
        allowed = ', '.join(Config.ALLOWED_EXTENSIONS)
        flash(f'Invalid file type. Allowed: {allowed}', 'error')
        return redirect(url_for('pages.documents_page'))
    
    filename = secure_filename(file.filename)
    
    # Process upload
    result = doc_service.upload_document(file, filename)
    
    # Handle result
    if result['success']:
        chunks = result.get('chunks_count', 0)
        flash(f'Successfully uploaded {filename}! Created {chunks} chunks.', 'success')
    else:
        error_msg = result.get('error', 'Unknown error')
        flash(f'Error processing file: {error_msg}', 'error')
    
    return redirect(url_for('pages.documents_page'))


@documents_bp.route('/search', methods=['POST'])
def search():
    """Search documents using semantic search"""
    doc_service = current_app.extensions.get('doc_service')
    
    if not doc_service:
        return create_error_response('Document processor not available', 500)
    
    query = request.form.get('query', '').strip()
    
    if not query:
        return create_error_response('Query is required')
    
    results = doc_service.search_documents(query, n_results=5)
    
    return create_success_response({
        'query': query,
        'results': results,
        'count': len(results)
    })


@documents_bp.route('/delete/<file_hash>', methods=['POST'])
def delete_document(file_hash: str):
    """Delete a document by file hash"""
    doc_service = current_app.extensions.get('doc_service')
    
    if not doc_service:
        flash('Document processor not available', 'error')
        return redirect(url_for('pages.documents_page'))
    
    if not file_hash:
        flash('Invalid document identifier', 'error')
        return redirect(url_for('pages.documents_page'))
    
    success = doc_service.delete_document(file_hash)
    
    if success:
        flash('Document deleted successfully', 'success')
    else:
        flash('Error deleting document', 'error')
    
    return redirect(url_for('pages.documents_page'))
