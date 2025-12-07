"""Conversation API routes blueprint"""
import logging
from flask import Blueprint, request, current_app
from utils import create_error_response, create_success_response

logger = logging.getLogger(__name__)

conversation_bp = Blueprint('conversation', __name__, url_prefix='/api/conversation')


@conversation_bp.route('/<string:conversation_id>/messages', methods=['POST'])
def create_message(conversation_id: str):
    """Handle conversation messages with RAG support"""
    rag_service = current_app.extensions.get('rag_service')
    
    if not rag_service:
        return create_error_response('Service not available', 500)
    
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
        response_data = rag_service.generate_response(user_message, use_rag, history)
        response_data['conversation_id'] = conversation_id
        
        return create_success_response(response_data)
        
    except Exception as e:
        logger.error(f"Conversation error: {str(e)}", exc_info=True)
        return create_error_response(f'Conversation failed: {str(e)}', 500)


@conversation_bp.route('/<string:conversation_id>/history', methods=['GET'])
def get_history(conversation_id: str):
    """Get conversation history (placeholder)"""
    logger.info(f"History request for conversation: {conversation_id}")
    # TODO: Implement proper session management with database
    return create_success_response({
        'conversation_id': conversation_id,
        'history': [],
        'message': 'Session management not yet implemented'
    })


@conversation_bp.route('/<string:conversation_id>/clear', methods=['POST'])
def clear_history(conversation_id: str):
    """Clear conversation history (placeholder)"""
    logger.info(f"Clear request for conversation: {conversation_id}")
    # TODO: Implement session clearing
    return create_success_response({
        'conversation_id': conversation_id,
        'message': 'Conversation cleared'
    })
