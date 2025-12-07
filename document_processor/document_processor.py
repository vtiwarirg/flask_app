import os
import warnings
import chromadb
from chromadb.config import Settings

# Suppress transformers warnings (we don't use local models)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', message='.*PyTorch.*TensorFlow.*Flax.*')

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import hashlib
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTextSplitter:
    """Simple text splitter for chunking documents"""
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text):
        """Split text into chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
            if start >= text_len:
                break
        
        return chunks


class DocumentProcessor:
    def __init__(self, persist_directory="./chroma_db", openai_api_key=None):
        """Initialize the document processor with ChromaDB and OpenAI embeddings"""
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in environment or pass as parameter.")
        
        # Initialize ChromaDB client with settings
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized at: {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
        
        # Get or create collection with distance metric
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents",
                metadata={
                    "description": "Document embeddings collection",
                    "embedding_model": "text-embedding-ada-002",
                    "hnsw:space": "cosine"  # Use cosine similarity
                }
            )
            logger.info(f"Collection 'documents' ready. Current count: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to create/get collection: {str(e)}")
            raise
        
        # Initialize OpenAI embeddings
        try:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-ada-002"
            )
            logger.info("OpenAI embeddings initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
            raise
        
        # Initialize text splitter with optimized settings
        self.text_splitter = SimpleTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from a TXT file using TextLoader"""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text = ""
        for doc in documents:
            text += doc.page_content + "\n"
        return text
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file using PyPDFLoader"""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text = ""
        for page in pages:
            text += page.page_content + "\n"
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file using Docx2txtLoader"""
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        text = ""
        for doc in documents:
            text += doc.page_content + "\n"
        return text
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text based on file type"""
        if file_type == 'txt':
            return self.extract_text_from_txt(file_path)
        elif file_type == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_type == 'docx':
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def generate_file_hash(self, file_path: str) -> str:
        """Generate a unique hash for the file"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def process_and_store_document(self, file_path: str, file_type: str, filename: str) -> Dict:
        """Process document and store in ChromaDB with embeddings"""
        try:
            logger.info(f"Processing document: {filename} ({file_type})")
            
            # Validate file exists
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f'File not found: {file_path}'
                }
            
            # Extract text
            text = self.extract_text(file_path, file_type)
            
            if not text.strip():
                logger.warning(f"No text content found in: {filename}")
                return {
                    'success': False,
                    'error': 'No text content found in document'
                }
            
            logger.info(f"Extracted {len(text)} characters from {filename}")
            
            # Generate file hash to avoid duplicates
            file_hash = self.generate_file_hash(file_path)
            logger.info(f"File hash: {file_hash}")
            
            # Check if document already exists
            existing_docs = self.collection.get(
                where={"file_hash": file_hash}
            )
            
            if existing_docs['ids']:
                logger.warning(f"Document already exists: {filename}")
                return {
                    'success': False,
                    'error': 'Document already exists in the database',
                    'file_hash': file_hash
                }
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Clean and validate chunks
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                cleaned_chunk = chunk.strip()
                if cleaned_chunk:  # Only keep non-empty chunks
                    valid_chunks.append(cleaned_chunk)
                    logger.debug(f"Chunk {i+1}/{len(chunks)}: {cleaned_chunk[:60]}...")
            
            if not valid_chunks:
                return {
                    'success': False,
                    'error': 'No valid chunks generated from document'
                }
            
            logger.info(f"Generated {len(valid_chunks)} valid chunks")
            
            # Generate embeddings for each chunk with batch processing
            logger.info("Generating embeddings...")
            try:
                chunk_embeddings = self.embeddings.embed_documents(valid_chunks)
                logger.info(f"Generated {len(chunk_embeddings)} embeddings")
                
                # Validate embeddings
                if len(chunk_embeddings) != len(valid_chunks):
                    raise ValueError(f"Embedding count mismatch: {len(chunk_embeddings)} vs {len(valid_chunks)}")
                
                # Validate embedding dimensions (OpenAI ada-002 = 1536 dimensions)
                if chunk_embeddings and len(chunk_embeddings[0]) != 1536:
                    logger.warning(f"Unexpected embedding dimension: {len(chunk_embeddings[0])}")
                    
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {str(e)}")
                return {
                    'success': False,
                    'error': f'Failed to generate embeddings: {str(e)}'
                }
            
            # Prepare data for ChromaDB
            ids = [f"{file_hash}_{i}" for i in range(len(valid_chunks))]
            metadatas = [
                {
                    "filename": filename,
                    "file_hash": file_hash,
                    "file_type": file_type,
                    "chunk_index": i,
                    "total_chunks": len(valid_chunks),
                    "chunk_size": len(chunk),
                    "embedding_model": "text-embedding-ada-002"
                }
                for i, chunk in enumerate(valid_chunks)
            ]
            
            # Add to ChromaDB with validation
            logger.info("Storing in ChromaDB...")
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=chunk_embeddings,
                    documents=valid_chunks,
                    metadatas=metadatas
                )
                logger.info(f"Successfully stored {len(valid_chunks)} chunks in ChromaDB")
                
                # Verify storage
                verification = self.collection.get(ids=[ids[0]])
                if not verification['ids']:
                    raise ValueError("Verification failed: chunks not found in database")
                
            except Exception as e:
                logger.error(f"Failed to store in ChromaDB: {str(e)}")
                return {
                    'success': False,
                    'error': f'Failed to store in database: {str(e)}'
                }
            
            # Return success with detailed info
            result = {
                'success': True,
                'filename': filename,
                'chunks_count': len(valid_chunks),
                'file_hash': file_hash,
                'total_characters': len(text),
                'embedding_dimensions': len(chunk_embeddings[0]) if chunk_embeddings else 0,
                'collection_total': self.collection.count()
            }
            
            logger.info(f"Document processed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error processing document: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents using semantic search with improved error handling"""
        try:
            if not query or not query.strip():
                logger.warning("Empty query provided")
                return []
            
            logger.info(f"Searching for: '{query}' (top {n_results} results)")
            
            # Check if collection has documents
            if self.collection.count() == 0:
                logger.warning("Collection is empty")
                return []
            
            # Generate embedding for the query
            try:
                query_embedding = self.embeddings.embed_query(query)
                logger.debug(f"Query embedding generated: {len(query_embedding)} dimensions")
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {str(e)}")
                return []
            
            # Search in ChromaDB
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, self.collection.count()),  # Don't exceed available docs
                    include=['documents', 'metadatas', 'distances']
                )
            except Exception as e:
                logger.error(f"ChromaDB query failed: {str(e)}")
                return []
            
            # Format results with enhanced information
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    # Calculate similarity score (1 - distance for cosine)
                    distance = results['distances'][0][i] if 'distances' in results and results['distances'] else None
                    similarity = (1 - distance) if distance is not None else None
                    
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': distance,
                        'similarity_score': round(similarity, 4) if similarity else None,
                        'rank': i + 1
                    })
                
                logger.info(f"Found {len(formatted_results)} results")
            else:
                logger.info("No results found")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}", exc_info=True)
            return []
    
    def get_all_documents(self) -> List[Dict]:
        """Get list of all unique documents in the database with enhanced metadata"""
        try:
            logger.info("Retrieving all documents from database")
            all_docs = self.collection.get()
            
            # Group by file_hash to get unique documents
            unique_docs = {}
            if all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    file_hash = metadata.get('file_hash')
                    if file_hash and file_hash not in unique_docs:
                        unique_docs[file_hash] = {
                            'filename': metadata.get('filename'),
                            'file_type': metadata.get('file_type'),
                            'total_chunks': metadata.get('total_chunks'),
                            'file_hash': file_hash,
                            'embedding_model': metadata.get('embedding_model', 'text-embedding-ada-002')
                        }
            
            result = list(unique_docs.values())
            logger.info(f"Found {len(result)} unique documents")
            return result
            
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}", exc_info=True)
            return []
    
    def delete_document(self, file_hash: str) -> bool:
        """Delete a document from the database with validation"""
        try:
            logger.info(f"Attempting to delete document: {file_hash}")
            
            # Get all chunks with this file_hash
            docs = self.collection.get(
                where={"file_hash": file_hash}
            )
            
            if docs['ids']:
                chunk_count = len(docs['ids'])
                self.collection.delete(ids=docs['ids'])
                logger.info(f"Deleted {chunk_count} chunks for document {file_hash}")
                
                # Verify deletion
                verification = self.collection.get(where={"file_hash": file_hash})
                if verification['ids']:
                    logger.error("Deletion verification failed")
                    return False
                
                return True
            else:
                logger.warning(f"No document found with hash: {file_hash}")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}", exc_info=True)
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get comprehensive statistics about the ChromaDB collection"""
        try:
            stats = {
                'total_chunks': self.collection.count(),
                'total_documents': len(self.get_all_documents()),
                'collection_name': self.collection.name,
                'persist_directory': self.persist_directory
            }
            
            # Get metadata statistics
            all_docs = self.collection.get()
            if all_docs['metadatas']:
                file_types = {}
                for metadata in all_docs['metadatas']:
                    ftype = metadata.get('file_type', 'unknown')
                    file_types[ftype] = file_types.get(ftype, 0) + 1
                stats['file_types'] = file_types
            
            logger.info(f"Collection stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
