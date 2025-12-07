# Quick Setup Guide

## 1. Create .env file

Create a file named `.env` in the root directory with the following content:

```
SECRET_KEY=your_secret_key_here_change_this
OPENAI_API_KEY=sk-your_openai_api_key_here
```

**Important:** Replace `sk-your_openai_api_key_here` with your actual OpenAI API key from https://platform.openai.com/api-keys

## 2. Run the Application

```powershell
python app.py
```

## 3. Access the Application

Open your browser and navigate to:

- Main page: http://127.0.0.1:5000
- Documents page: http://127.0.0.1:5000/documents

## Features

### Upload Documents

1. Go to the Documents page
2. Click "Choose a document" button
3. Select a TXT, PDF, or DOCX file
4. Click "Upload & Process"
5. The document will be embedded and stored in ChromaDB

### Search Documents

1. Enter a natural language query in the search box
2. Click "Search" button
3. View relevant document chunks with similarity scores

### Delete Documents

1. Find the document in the list
2. Click the red "Delete" button
3. Confirm the deletion

## Technical Details

- Documents are split into 1000-character chunks with 200-character overlap
- OpenAI's text-embedding-ada-002 model is used for embeddings
- ChromaDB stores the vectors locally in the `chroma_db` folder
- Duplicate detection using MD5 file hashing
- Maximum file size: 16MB

## Troubleshooting

### Missing OpenAI API Key

If you see authentication errors, make sure your `.env` file contains a valid OpenAI API key.

### Import Errors

If you see import errors, make sure all packages are installed:

```powershell
pip install -r requirements.txt
```

### ChromaDB Errors

If ChromaDB gives errors, try deleting the `chroma_db` folder and restarting the app.
