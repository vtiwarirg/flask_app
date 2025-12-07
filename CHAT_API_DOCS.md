# Chat & Conversation API Documentation

## Overview

The chat interface provides a RAG (Retrieval Augmented Generation) powered conversation system that searches through uploaded documents to provide contextual answers.

## Features

### 🎯 **Chat Interface** (`/chat`)

- Interactive web-based chat UI
- Real-time document search integration
- Source citation and similarity scores
- Conversation history tracking
- Quick action buttons for common queries

### 🔌 **API Endpoints**

#### **POST /api/conversation**

Main conversation endpoint with RAG support.

**Request:**

```json
{
  "message": "What are the company policies?",
  "use_rag": true,
  "history": []
}
```

**Response:**

```json
{
  "success": true,
  "message": "What are the company policies?",
  "response": "Based on the documents, the company policies include...",
  "sources": [
    {
      "filename": "policy.pdf",
      "chunk_index": 5,
      "similarity": 0.89,
      "content_preview": "Company policies state that..."
    }
  ],
  "context_used": true,
  "use_rag": true
}
```

**Parameters:**

- `message` (string, required): User's question
- `use_rag` (boolean, optional, default: true): Enable document search
- `history` (array, optional): Conversation history for context

**Response Fields:**

- `success`: Operation status
- `response`: AI-generated answer
- `sources`: Array of source documents used
- `context_used`: Whether RAG found relevant context
- `similarity`: Similarity score (0-1) for each source

#### **GET /api/conversation/history**

Retrieve conversation history (placeholder for future implementation).

**Response:**

```json
{
  "success": true,
  "history": [],
  "message": "Session management not yet implemented"
}
```

#### **POST /api/conversation/clear**

Clear conversation history (placeholder).

**Response:**

```json
{
  "success": true,
  "message": "Conversation cleared"
}
```

## How RAG Works

### 1. **Document Upload**

- User uploads documents (PDF, DOCX, TXT)
- Documents are chunked into 1000-character segments
- Each chunk is embedded using OpenAI's text-embedding-ada-002
- Embeddings stored in ChromaDB

### 2. **Query Processing**

```
User Question → Embedding Generation → Vector Search → Top K Results
```

### 3. **Context Retrieval**

- Query is embedded using the same model
- ChromaDB performs cosine similarity search
- Top 3 most relevant chunks retrieved
- Chunks include source metadata (filename, page, etc.)

### 4. **Response Generation**

```
User Question + Retrieved Context → OpenAI GPT-3.5-turbo → Answer with Citations
```

**Two Modes:**

**With OpenAI API:**

```python
System Prompt: "You are a helpful assistant that answers based on context..."
User Prompt: "Context: [Retrieved chunks]\n\nQuestion: [User query]"
→ GPT-3.5-turbo generates contextual answer
```

**Without OpenAI API (Fallback):**

```python
→ Returns formatted context chunks directly
```

## Usage Examples

### Example 1: Basic Question

```javascript
fetch("/api/conversation", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "What is the company vacation policy?",
    use_rag: true,
  }),
});
```

**Response:**

> "According to the employee handbook, the company offers 15 days of paid vacation per year. Employees accrue 1.25 days per month..."
>
> **Sources:** employee_handbook.pdf (Similarity: 92%)

### Example 2: Without RAG

```javascript
fetch("/api/conversation", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message: "Hello!",
    use_rag: false,
  }),
});
```

**Response:**

> "You said: Hello! (RAG is disabled. Enable it to search documents.)"

### Example 3: No Relevant Documents

```javascript
// Query about topic not in documents
{
  "message": "What is quantum computing?",
  "use_rag": true
}
```

**Response:**

> "I couldn't find any relevant information in the uploaded documents. Could you please rephrase your question or upload relevant documents?"

## Web Interface Features

### Chat Window

- **User Messages**: Blue background, right-aligned
- **AI Responses**: Gray background, left-aligned
- **Loading Indicator**: Shows during search and generation
- **Auto-scroll**: Automatically scrolls to latest message

### Sidebar Components

**1. Database Stats**

- Total documents count
- Total chunks count
- File type breakdown

**2. Sources Panel**

- Shows sources used in last response
- Displays similarity scores
- Preview of relevant content
- Links to source documents

**3. Quick Actions**

- Pre-defined question templates
- One-click query insertion

### Settings

- **RAG Toggle**: Enable/disable document search
- **Clear Chat**: Reset conversation
- **Manage Documents**: Link to upload page

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional
SECRET_KEY=your-flask-secret
```

### RAG Parameters (in document_processor.py)

```python
chunk_size = 1000          # Characters per chunk
chunk_overlap = 200        # Overlap between chunks
n_results = 3              # Top K documents to retrieve
embedding_model = "text-embedding-ada-002"
chat_model = "gpt-3.5-turbo"
temperature = 0.7          # Response creativity (0-1)
max_tokens = 500          # Max response length
```

## Error Handling

### Common Errors

**1. No Documents Uploaded**

```json
{
  "response": "I couldn't find any relevant information...",
  "sources": [],
  "context_used": false
}
```

**2. Document Processor Not Initialized**

```json
{
  "success": false,
  "error": "Document processor not initialized"
}
```

**3. OpenAI API Error**

```json
{
  "response": "Based on the documents, here are the most relevant excerpts...\n(Note: AI generation unavailable - Rate limit exceeded)"
}
```

**4. Empty Query**

```json
{
  "success": false,
  "error": "Message is required"
}
```

## Performance

### Response Times

- **Vector Search**: < 100ms (for < 10,000 chunks)
- **Embedding Generation**: ~200ms per query
- **OpenAI GPT Response**: 1-3 seconds
- **Total**: 1.5-3.5 seconds typical

### Optimization Tips

1. Keep database size < 100,000 chunks
2. Use specific queries for better results
3. Upload relevant documents only
4. Monitor OpenAI API usage

## Security Considerations

1. **API Key Protection**: Never expose in frontend
2. **Input Validation**: Sanitize all user inputs
3. **Rate Limiting**: Implement for production
4. **Session Management**: Add user authentication
5. **HTTPS**: Required for production deployment

## Future Enhancements

- [ ] Persistent conversation history
- [ ] Multi-user session management
- [ ] Conversation export (PDF, TXT)
- [ ] Advanced filtering (by document, date, etc.)
- [ ] Custom system prompts
- [ ] Multiple AI model support
- [ ] Voice input/output
- [ ] Multi-language support
- [ ] Conversation analytics

## Troubleshooting

### Chat not responding

1. Check OpenAI API key is configured
2. Verify documents are uploaded
3. Check browser console for errors
4. Review Flask logs

### Poor answer quality

1. Upload more relevant documents
2. Use more specific questions
3. Check chunk size settings
4. Verify embedding quality

### Slow responses

1. Reduce n_results parameter
2. Optimize database size
3. Check OpenAI API status
4. Monitor server resources

## API Testing

### Using cURL

```bash
# Test conversation endpoint
curl -X POST http://localhost:5000/api/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is in the documents?",
    "use_rag": true
  }'
```

### Using Python

```python
import requests

response = requests.post('http://localhost:5000/api/conversation', json={
    'message': 'What are the key points?',
    'use_rag': True
})

print(response.json())
```

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
