# Flash Card System with RAG-powered FAQ

## Overview
This project is a Flash Card System for learning English vocabulary, enhanced with a Retrieval-Augmented Generation (RAG) powered FAQ system that provides intelligent answers to user questions about the system.

## Features

### Core Features
- Vocabulary flashcards with pronunciation
- Text analysis for vocabulary extraction
- Audio generation for pronunciation
- Chat interface for interaction
- History management

### New RAG FAQ Feature
The FAQ system uses modern RAG architecture to provide accurate and context-aware answers to user questions about the system.

## Solution Approach

### 1. RAG Implementation
- **Document Loading**: FAQ data is stored in `faq.json` and loaded as documents with metadata
- **Embedding Generation**: Using Azure OpenAI embeddings (text-embedding-3-small model)
- **Vector Storage**: ChromaDB for efficient vector storage and retrieval
- **LLM Integration**: Azure OpenAI GPT-4 for generating contextual answers

### 2. Architecture Components

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  FAQ.json   │────>│ ChromaDB     │────>│   Azure    │
│  Documents  │     │ Vector Store │     │  OpenAI    │
└─────────────┘     └──────────────┘     └────────────┘
       │                   │                    │
       └───────────────────┴────────────────────┘
                          │
                    ┌──────────────┐
                    │    Flask     │
                    │   Backend    │
                    └──────────────┘
                          │
                    ┌──────────────┐
                    │  Web UI      │
                    │  Interface   │
                    └──────────────┘
```

### 3. Implementation Steps

1. **Vector Store Setup**
   ```python
   def setup_faq_vectorstore():
       # Initialize embeddings with Azure OpenAI
       embedding_function = AzureOpenAIEmbeddings(...)
       
       # Load FAQ documents
       docs = load_faq_documents()
       
       # Create vector store
       vectorstore = Chroma.from_documents(...)
       
       return vectorstore
   ```

2. **RAG Query Processing**
   ```python
   def rag_faq_answer(query):
       # Retrieve relevant documents
       docs = vectorstore.similarity_search(query)
       
       # Generate context-aware answer
       llm_response = chain.invoke({
           "context": context,
           "question": query
       })
       
       return response
   ```

3. **API Integration**
   - RESTful endpoint `/api/faq` for handling FAQ queries
   - JSON response format with answer and success status
   - Frontend integration with async requests

## Technical Challenges & Solutions

1. **ChromaDB Integration**
   - **Challenge**: Deprecated imports and API changes in LangChain
   - **Solution**: Updated to use `langchain_chroma` package and new API format

2. **Azure OpenAI Authentication**
   - **Challenge**: Model access restrictions and authentication errors
   - **Solution**: Properly configured deployment names and model specifications
   ```python
   embeddings = AzureOpenAIEmbeddings(
       deployment="text-embedding-3-small",
       model="text-embedding-3-small"
   )
   ```

3. **Vector Storage Persistence**
   - **Challenge**: Maintaining vector store state between app restarts
   - **Solution**: Implemented persistent storage with ChromaDB
   ```python
   vectorstore = Chroma(
       persist_directory=CHROMA_PERSIST_DIR
   )
   ```

4. **UI Consistency**
   - **Challenge**: Maintaining consistent UI/UX across the application
   - **Solution**: Standardized templates and styling components

## Environment Setup

Required environment variables:
```env
AZURE_OPENAI_EMBEDDING_ENDPOINT=<your-endpoint>
AZURE_OPENAI_EMBEDDING_API_KEY=<your-api-key>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

AZURE_OPENAI_LLM_API_KEY=<your-api-key>
AZURE_OPENAI_LLM_ENDPOINT=<your-endpoint>
AZURE_OPENAI_LLM_MODEL=GPT-4o-mini
```

## Dependencies

```plaintext
langchain>=0.1.5
langchain-openai>=0.0.5
langchain-community>=0.0.13
langchain-chroma>=0.0.5
chromadb>=0.4.22
openai>=1.6.0
```

## Running the Application

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the FAQ system at `http://localhost:5000/faq`

## Future Improvements

1. Enhanced Answer Generation
   - Implement answer ranking
   - Add source citations
   - Support follow-up questions

2. Performance Optimizations
   - Implement caching for frequent queries
   - Batch embedding processing
   - Optimize vector search parameters

3. UI Enhancements
   - Add loading states
   - Implement answer highlighting
   - Add feedback mechanism

4. System Expansion
   - Support multiple languages
   - Add dynamic FAQ updates
   - Implement user feedback loop for answer quality
