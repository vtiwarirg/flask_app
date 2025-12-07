import logging
import os
from typing import Dict, Any, Optional
from werkzeug.utils import secure_filename
from config import Config

logger = logging.getLogger(__name__)


class DocumentService:
   
    def __init__(self, doc_processor):
        self.doc_processor = doc_processor
    
    def upload_document(self, file, filename: str) -> Dict[str, Any]:
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        try:
            # Save file temporarily
            file.save(file_path)
            logger.info(f"File saved: {filename}")
            
            # Get file extension
            file_type = filename.rsplit('.', 1)[1].lower()
            
            # Process and store document
            result = self.doc_processor.process_and_store_document(
                file_path, file_type, filename
            )
            
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Temporary file removed: {filename}")
            
            return result
            
        except Exception as e:
            logger.error(f"Upload error for {filename}: {str(e)}")
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_document(self, file_hash: str) -> bool:
        try:
            success = self.doc_processor.delete_document(file_hash)
            if success:
                logger.info(f"Document deleted: {file_hash}")
            else:
                logger.warning(f"Failed to delete document: {file_hash}")
            return success
        except Exception as e:
            logger.error(f"Delete error for {file_hash}: {str(e)}")
            return False
    
    def get_all_documents(self) -> list:
        try:
            return self.doc_processor.get_all_documents()
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return []
    
    def search_documents(self, query: str, n_results: int = 5) -> list:
        try:
            results = self.doc_processor.search_documents(query, n_results=n_results)
            logger.info(f"Search query: '{query}' - Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            return self.doc_processor.get_collection_stats()
        except Exception as e:
            logger.warning(f"Failed to get collection stats: {str(e)}")
            return {'total_documents': 0, 'total_chunks': 0}
