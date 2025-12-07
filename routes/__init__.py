"""Routes package"""
from .pages import pages_bp
from .documents import documents_bp
from .conversation import conversation_bp

__all__ = ['pages_bp', 'documents_bp', 'conversation_bp']
