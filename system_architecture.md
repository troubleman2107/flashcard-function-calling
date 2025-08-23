# Flashcard System Architecture Diagram

## System Overview
This document provides a comprehensive system architecture diagram for the AI-powered English Vocabulary Learning Flashcard System.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Web UI - HTML/CSS/JS]
        UI --> |HTTP Requests| API[Flask API Layer]
    end
    
    subgraph "Backend Layer"
        API --> |Route to| FLASK[Flask Application]
        FLASK --> |Initialize| COMPONENTS[Core Components]
    end
    
    subgraph "AI/ML Layer"
        COMPONENTS --> |Uses| LANGCHAIN[LangChain Framework]
        COMPONENTS --> |Uses| OPENAI[Azure OpenAI API]
        COMPONENTS --> |Uses| TTS[Text-to-Speech Service]
        COMPONENTS --> |Uses| EMBEDDINGS[Azure OpenAI Embeddings]
    end
    
    subgraph "Data Layer"
        COMPONENTS --> |Stores| CHROMADB[ChromaDB Vector Database]
        COMPONENTS --> |Stores| HISTORY[History JSON Files]
        COMPONENTS --> |Stores| AUDIO[Audio Files]
        COMPONENTS --> |Stores| FAQ[FAQ Vector Store]
    end
    
    subgraph "External Services"
        OPENAI --> |API Calls| AZURE[Azure OpenAI Services]
        EMBEDDINGS --> |API Calls| AZURE
        TTS --> |Model Loading| HF[Hugging Face Models]
    end
```

## Detailed Component Architecture

```mermaid
graph TB
    subgraph "User Interface"
        direction TB
        MAIN[Main Page - /]
        CHAT[Chat Interface - /chat]
        SEARCH[Search Page - /search]
        LIST[Flashcard List - /list]
        FAQ[FAQ Page - /faq]
    end
    
    subgraph "API Endpoints"
        direction TB
        API_MAIN[POST / - Add Word]
        API_CHAT[POST /api/chat - Process Chat]
        API_SEARCH[POST /api/semantic-search - Search]
        API_DELETE[POST /api/delete-word - Delete Word]
        API_CLEAR[POST /api/clear-all-data - Clear Data]
        API_CATEGORY[POST /api/search-category - Category Search]
        API_CATEGORIES[GET /api/categories - Get Categories]
        API_WORD[GET /api/word/<word> - Get Word Info]
        API_AUDIO[GET /api/generate-audio/<word> - Generate Audio]
        API_FAQ[POST /api/faq - FAQ Query]
    end
    
    subgraph "Core Business Logic"
        direction TB
        VOCAB_MGR[Vocabulary Manager]
        TTS_SERVICE[TTS Service]
        LANGCHAIN_AGENT[LangChain Agent]
        WORD_ANALYZER[Word Analyzer]
        TEXT_ANALYZER[Text Analyzer]
        SEARCH_ENGINE[Search Engine]
        FAQ_ENGINE[FAQ RAG Engine]
    end
    
    subgraph "Data Models"
        direction TB
        VOCAB_WORD[VocabularyWord Model]
        VOCAB_LIST[VocabularyList Model]
        SEARCH_RESULT[SemanticSearchResult Model]
        FAQ_DOC[FAQ Document Model]
    end
    
    subgraph "Data Storage"
        direction TB
        CHROMADB_COLL[ChromaDB Collection]
        HISTORY_JSON[History.json]
        AUDIO_FILES[Static/Audio Files]
        FAQ_VECTOR[FAQ Vector Store]
    end
    
    subgraph "External AI Services"
        direction TB
        AZURE_LLM[Azure OpenAI LLM]
        AZURE_EMB[Azure OpenAI Embeddings]
        HF_TTS[Hugging Face TTS Models]
    end
    
    subgraph "LangChain Components"
        direction TB
        LLM[ChatOpenAI]
        MEMORY[Conversation Memory]
        TOOLS[LangChain Tools]
        PROMPTS[Prompt Templates]
        PARSERS[Output Parsers]
        AGENTS[AI Agents]
    end
    
    %% User Interface to API connections
    MAIN --> API_MAIN
    CHAT --> API_CHAT
    SEARCH --> API_SEARCH
    LIST --> API_DELETE
    LIST --> API_CLEAR
    SEARCH --> API_CATEGORY
    SEARCH --> API_CATEGORIES
    MAIN --> API_WORD
    MAIN --> API_AUDIO
    FAQ --> API_FAQ
    
    %% API to Business Logic connections
    API_MAIN --> WORD_ANALYZER
    API_CHAT --> LANGCHAIN_AGENT
    API_SEARCH --> SEARCH_ENGINE
    API_DELETE --> VOCAB_MGR
    API_CLEAR --> VOCAB_MGR
    API_CATEGORY --> VOCAB_MGR
    API_CATEGORIES --> VOCAB_MGR
    API_WORD --> WORD_ANALYZER
    API_AUDIO --> TTS_SERVICE
    API_FAQ --> FAQ_ENGINE
    
    %% Business Logic to Data Models
    WORD_ANALYZER --> VOCAB_WORD
    TEXT_ANALYZER --> VOCAB_LIST
    SEARCH_ENGINE --> SEARCH_RESULT
    FAQ_ENGINE --> FAQ_DOC
    
    %% Business Logic to Data Storage
    VOCAB_MGR --> CHROMADB_COLL
    VOCAB_MGR --> HISTORY_JSON
    TTS_SERVICE --> AUDIO_FILES
    FAQ_ENGINE --> FAQ_VECTOR
    
    %% Business Logic to External Services
    WORD_ANALYZER --> AZURE_LLM
    TEXT_ANALYZER --> AZURE_LLM
    LANGCHAIN_AGENT --> AZURE_LLM
    SEARCH_ENGINE --> AZURE_EMB
    FAQ_ENGINE --> AZURE_EMB
    TTS_SERVICE --> HF_TTS
    
    %% LangChain Component connections
    LANGCHAIN_AGENT --> LLM
    LANGCHAIN_AGENT --> MEMORY
    LANGCHAIN_AGENT --> TOOLS
    LANGCHAIN_AGENT --> PROMPTS
    LANGCHAIN_AGENT --> PARSERS
    LANGCHAIN_AGENT --> AGENTS
```

## Data Flow Architecture

```mermaid
flowchart TD
    subgraph "User Input Flow"
        USER[User Input] --> INPUT_TYPE{Input Type?}
        INPUT_TYPE -->|Single Word| WORD_FLOW[Word Analysis Flow]
        INPUT_TYPE -->|Text Passage| TEXT_FLOW[Text Analysis Flow]
        INPUT_TYPE -->|Search Query| SEARCH_FLOW[Search Flow]
        INPUT_TYPE -->|Chat Message| CHAT_FLOW[Chat Flow]
        INPUT_TYPE -->|FAQ Question| FAQ_FLOW[FAQ Flow]
    end
    
    subgraph "Word Analysis Flow"
        WORD_FLOW --> CHECK_EXISTS{Word Exists?}
        CHECK_EXISTS -->|Yes| RETURN_EXISTING[Return Existing Data]
        CHECK_EXISTS -->|No| ANALYZE_WORD[Analyze with LangChain]
        ANALYZE_WORD --> CREATE_STRUCTURE[Create Structured Data]
        CREATE_STRUCTURE --> CLASSIFY_CATEGORY[Classify Category]
        CLASSIFY_CATEGORY --> GENERATE_AUDIO[Generate TTS Audio]
        GENERATE_AUDIO --> SAVE_CHROMADB[Save to ChromaDB]
        SAVE_CHROMADB --> SAVE_HISTORY[Save to History]
        SAVE_HISTORY --> RETURN_RESULT[Return Result]
    end
    
    subgraph "Text Analysis Flow"
        TEXT_FLOW --> EXTRACT_WORDS[Extract Vocabulary Words]
        EXTRACT_WORDS --> ANALYZE_EACH[Analyze Each Word]
        ANALYZE_EACH --> SAVE_ALL[Save All Words]
        SAVE_ALL --> RETURN_COUNT[Return Count & Results]
    end
    
    subgraph "Search Flow"
        SEARCH_FLOW --> CHROMADB_SEARCH[ChromaDB Semantic Search]
        CHROMADB_SEARCH --> HAS_RESULTS{Has Results?}
        HAS_RESULTS -->|Yes| RETURN_CHROMADB[Return ChromaDB Results]
        HAS_RESULTS -->|No| OPENAI_GENERATE[OpenAI Generate Words]
        OPENAI_GENERATE --> KEYWORD_FALLBACK[Keyword Fallback Search]
        KEYWORD_FALLBACK --> RETURN_FINAL[Return Final Results]
    end
    
    subgraph "Chat Flow"
        CHAT_FLOW --> AGENT_PROCESS[LangChain Agent Process]
        AGENT_PROCESS --> TOOL_SELECTION{Tool Needed?}
        TOOL_SELECTION -->|Yes| EXECUTE_TOOL[Execute Tool]
        TOOL_SELECTION -->|No| GENERATE_RESPONSE[Generate Response]
        EXECUTE_TOOL --> GENERATE_RESPONSE
        GENERATE_RESPONSE --> RETURN_CHAT[Return Chat Response]
    end
    
    subgraph "FAQ Flow"
        FAQ_FLOW --> VECTOR_SEARCH[Vector Store Search]
        VECTOR_SEARCH --> RAG_GENERATE[RAG Answer Generation]
        RAG_GENERATE --> RETURN_FAQ[Return FAQ Answer]
    end
```

## Technology Stack Architecture

```mermaid
graph TB
    subgraph "Frontend Technologies"
        HTML[HTML5]
        CSS[CSS3]
        JS[JavaScript]
        JINJA[Jinja2 Templates]
    end
    
    subgraph "Backend Framework"
        FLASK[Flask 3.0.0]
        PYTHON[Python 3.x]
        DOTENV[python-dotenv]
    end
    
    subgraph "AI/ML Technologies"
        LANGCHAIN[LangChain Framework]
        OPENAI[OpenAI API 1.6.1]
        TRANSFORMERS[Transformers 4.36.2]
        TORCH[PyTorch 2.8.0]
        SOUNDFILE[soundfile 0.12.1]
    end
    
    subgraph "Database & Storage"
        CHROMADB[ChromaDB 1.0.20]
        JSON[JSON Files]
        VECTOR[Vector Embeddings]
    end
    
    subgraph "Cloud Services"
        AZURE_OPENAI[Azure OpenAI]
        AZURE_EMBEDDINGS[Azure OpenAI Embeddings]
        HF_MODELS[Hugging Face Models]
    end
    
    subgraph "Data Processing"
        PYDANTIC[Pydantic 2.11.7]
        NUMPY[NumPy 2.0+]
        HASH[Hashlib]
        RE[Regex]
    end
    
    HTML --> FLASK
    CSS --> FLASK
    JS --> FLASK
    JINJA --> FLASK
    
    FLASK --> PYTHON
    PYTHON --> DOTENV
    
    PYTHON --> LANGCHAIN
    PYTHON --> OPENAI
    PYTHON --> TRANSFORMERS
    PYTHON --> TORCH
    PYTHON --> SOUNDFILE
    
    LANGCHAIN --> AZURE_OPENAI
    OPENAI --> AZURE_OPENAI
    TRANSFORMERS --> HF_MODELS
    TORCH --> HF_MODELS
    
    PYTHON --> CHROMADB
    PYTHON --> JSON
    PYTHON --> VECTOR
    
    PYTHON --> PYDANTIC
    PYTHON --> NUMPY
    PYTHON --> HASH
    PYTHON --> RE
```

## Security & Configuration Architecture

```mermaid
graph TB
    subgraph "Environment Configuration"
        ENV[Environment Variables]
        ENV --> AZURE_LLM_ENDPOINT[AZURE_OPENAI_LLM_ENDPOINT]
        ENV --> AZURE_LLM_KEY[AZURE_OPENAI_LLM_API_KEY]
        ENV --> AZURE_EMB_ENDPOINT[AZURE_OPENAI_EMBEDDING_ENDPOINT]
        ENV --> AZURE_EMB_KEY[AZURE_OPENAI_EMBEDDING_API_KEY]
        ENV --> AZURE_EMB_MODEL[AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME]
        ENV --> AZURE_LLM_MODEL[AZURE_OPENAI_LLM_MODEL]
    end
    
    subgraph "Security Measures"
        API_KEYS[API Key Management]
        ENDPOINT_SECURITY[Endpoint Security]
        DATA_VALIDATION[Input Validation]
        ERROR_HANDLING[Error Handling]
    end
    
    subgraph "Data Protection"
        CHROMA_SECURITY[ChromaDB Security]
        FILE_PERMISSIONS[File Permissions]
        AUDIO_SECURITY[Audio File Security]
    end
    
    ENV --> API_KEYS
    ENV --> ENDPOINT_SECURITY
    
    API_KEYS --> DATA_VALIDATION
    ENDPOINT_SECURITY --> ERROR_HANDLING
    
    DATA_VALIDATION --> CHROMA_SECURITY
    ERROR_HANDLING --> FILE_PERMISSIONS
    FILE_PERMISSIONS --> AUDIO_SECURITY
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Local Development]
        DEV --> DEV_PORT[Port 5000]
        DEV --> DEV_DEBUG[Debug Mode]
    end
    
    subgraph "Production Environment"
        PROD[Production Server]
        PROD --> PROD_PORT[Environment PORT]
        PROD --> PROD_HOST[Host 0.0.0.0]
    end
    
    subgraph "File Structure"
        STATIC[Static Files]
        TEMPLATES[Templates]
        AUDIO[Audio Directory]
        CHROMA[ChromaDB Directory]
        LOGS[Log Files]
    end
    
    DEV --> STATIC
    DEV --> TEMPLATES
    DEV --> AUDIO
    DEV --> CHROMA
    
    PROD --> STATIC
    PROD --> TEMPLATES
    PROD --> AUDIO
    PROD --> CHROMA
    PROD --> LOGS
```

## Key Features & Capabilities

- **AI-Powered Vocabulary Analysis**: Uses LangChain and OpenAI for intelligent word analysis
- **Semantic Search**: ChromaDB vector database with fallback mechanisms
- **Text-to-Speech**: Hugging Face TTS models for pronunciation
- **Intelligent Categorization**: Automatic category classification using AI
- **Multi-language Support**: Vietnamese and English interface
- **Conversational AI**: LangChain agents for natural language interaction
- **RAG System**: FAQ system with retrieval-augmented generation
- **Real-time Processing**: Immediate response to user queries
- **Scalable Architecture**: Modular design for easy extension
- **Error Handling**: Robust fallback mechanisms and error recovery

## Performance Characteristics

- **Response Time**: < 2 seconds for most operations
- **Concurrent Users**: Supports multiple simultaneous users
- **Data Persistence**: ChromaDB for vector storage, JSON for metadata
- **Caching**: Audio file caching to avoid regeneration
- **Memory Management**: Efficient conversation memory with windowing
- **Scalability**: Horizontal scaling capability through modular design
