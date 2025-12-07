"""Helper utility functions"""
from flask import jsonify
from typing import Dict, Any, Tuple
from config import Config


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def create_error_response(message: str, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
    """Create standardized error response
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Tuple of (JSON response, status code)
    """
    return jsonify({
        'success': False,
        'error': message
    }), status_code


def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized success response
    
    Args:
        data: Data to include in response
        
    Returns:
        JSON response with success flag
    """
    response = {'success': True}
    response.update(data)
    return jsonify(response)
