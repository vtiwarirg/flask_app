"""Application configuration module"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('FLASK_HOST', '127.0.0.1')
    PORT = int(os.getenv('FLASK_PORT', '5000'))
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
    
    # Database settings
    CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
    
    # RAG settings
    SEARCH_RESULTS_LIMIT = int(os.getenv('SEARCH_RESULTS_LIMIT', '10'))
    HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', '10'))
    MIN_SIMILARITY_THRESHOLD = float(os.getenv('MIN_SIMILARITY_THRESHOLD', '0.25'))  # Filter low-relevance results
    
    @staticmethod
    def validate():
        """Validate critical configuration"""
        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == 'your_openai_api_key_here':
            return False, "OpenAI API key not configured"
        return True, "Configuration valid"
