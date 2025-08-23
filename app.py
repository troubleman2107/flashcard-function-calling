from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    send_from_directory,
)
from transformers import VitsModel, AutoTokenizer
import openai
import json
import os
import re
import torch
import soundfile as sf
import time
import hashlib
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# LangChain imports
from langchain_openai import ChatOpenAI, AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

app = Flask(__name__)


# Pydantic models for structured output
class VocabularyWord(BaseModel):
    word: str = Field(
        description="The English word being analyzed, fix this word if it is not a valid English word"
    )
    vietnamese_meaning: str = Field(description="The meaning of the word in Vietnamese")
    part_of_speech: str = Field(
        description="Part of speech (noun, verb, adjective, etc.)"
    )
    phonetic: Optional[str] = Field(description="Phonetic pronunciation of the word")
    example_sentences: List[str] = Field(
        description="Two example sentences using the word", min_items=2, max_items=2
    )
    mnemonic_tip: str = Field(
        description="A memorable tip or mnemonic to help learn the word"
    )
    difficulty_level: str = Field(
        description="Difficulty level of the word",
        pattern="^(beginner|intermediate|advanced)$",
    )
    synonyms: List[str] = Field(description="List of synonyms (up to 3)", max_items=3)


class VocabularyList(BaseModel):
    vocabulary_list: List[VocabularyWord] = Field(
        description="List of vocabulary words extracted from the text",
        min_items=1,
        max_items=10,
    )


class SemanticSearchResult(BaseModel):
    success: bool
    message: str
    query: str
    results: List[dict]
    count: Optional[int] = 0


# LangChain setup
def setup_langchain():
    """Initialize LangChain components"""
    llm = ChatOpenAI(
        openai_api_base=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
        model_name="GPT-4o-mini",
        temperature=0.1,
    )

    # Memory for conversation context
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True, k=5  # Keep last 5 exchanges
    )

    return llm, memory


# Initialize LangChain
llm, memory = setup_langchain()


# TTS Service Class (unchanged)
class TTSService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.audio_dir = "static/audio"
        os.makedirs(self.audio_dir, exist_ok=True)
        self._initialize_models()

    def _initialize_models(self):
        """Initialize TTS models lazily"""
        try:
            print("Loading TTS models...")
            self.model = VitsModel.from_pretrained("facebook/mms-tts-eng")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            print("TTS models loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load TTS models: {e}")
            print(
                "TTS functionality will be disabled. The app will continue to work without audio generation."
            )
            self.model = None
            self.tokenizer = None

    def generate_audio(self, word, vietnamese_meaning):
        """Generate audio for a word with Vietnamese meaning"""
        if not self.model or not self.tokenizer:
            print("TTS models not available")
            return None

        try:
            tts_text = f"{word}"
            content_hash = hashlib.md5(tts_text.encode()).hexdigest()[:8]
            filename = f"{word}_{content_hash}.wav"
            filepath = os.path.join(self.audio_dir, filename)

            if os.path.exists(filepath):
                return f"/static/audio/{filename}"

            inputs = self.tokenizer(tts_text, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**inputs).waveform

            sf.write(
                filepath, output.squeeze().numpy(), self.model.config.sampling_rate
            )
            print(f"Generated audio: {filepath}")

            return f"/static/audio/{filename}"

        except Exception as e:
            print(f"Error generating audio for '{word}': {e}")
            return None


# Initialize TTS service
tts_service = TTSService()


# Vocabulary Management with ChromaDB (enhanced with LangChain)
class VocabularyManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="vocabulary", metadata={"hnsw:space": "cosine"}
        )

    def classify_category(
        self, word, vietnamese_meaning, part_of_speech, example_sentences
    ):
        """Classify category using LangChain"""
        try:
            context = f"Word: {word}\nMeaning: {vietnamese_meaning}\nPart of speech: {part_of_speech}\nExamples: {' '.join(example_sentences)}"

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a vocabulary categorization expert. "
                        "Classify the given English word into ONE of these categories: "
                        "Business, Technology, Education, Health, Travel, Food, Sports, "
                        "Entertainment, Science, Art, Nature, Family, Emotions, Time, "
                        "Colors, Numbers, Animals, Transportation, Clothing, Weather. "
                        "Return ONLY the category name, nothing else.",
                    ),
                    ("user", "Classify this word: {context}"),
                ]
            )

            chain = prompt | llm
            response = chain.invoke({"context": context})

            category = response.content.strip()
            return category

        except Exception as e:
            print(f"Error classifying category: {e}")
            return "General"

    def add_vocabulary(self, word_data):
        """Add vocabulary to ChromaDB with automatic category"""
        try:
            word = word_data.get("word", "")
            vietnamese_meaning = word_data.get("vietnamese_meaning", "")
            part_of_speech = word_data.get("part_of_speech", "")
            example_sentences = word_data.get("example_sentences", [])

            category = self.classify_category(
                word, vietnamese_meaning, part_of_speech, example_sentences
            )

            document_text = f"{word} {vietnamese_meaning} {part_of_speech} {' '.join(example_sentences)}"

            metadata = {
                "word": word,
                "vietnamese_meaning": vietnamese_meaning,
                "part_of_speech": part_of_speech,
                "category": category,
                "difficulty_level": word_data.get("difficulty_level", "intermediate"),
                "phonetic": word_data.get("phonetic", ""),
                "synonyms": ",".join(word_data.get("synonyms", [])),
                "mnemonic_tip": word_data.get("mnemonic_tip", ""),
                "example_sentences": "|".join(example_sentences),
            }

            self.collection.add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[f"word_{word.lower()}_{int(time.time())}"],
            )

            print(f"Added word '{word}' to category '{category}'")
            return category

        except Exception as e:
            print(f"Error adding vocabulary to ChromaDB: {e}")
            return None

    def search_by_category(self, category, limit=50):
        """Search vocabulary by specific category"""
        try:
            all_data = self.collection.get(include=["metadatas"])
            results = []

            if all_data["metadatas"]:
                for metadata in all_data["metadatas"]:
                    if metadata.get("category", "General").lower() == category.lower():
                        results.append(
                            {
                                "word": metadata["word"],
                                "vietnamese_meaning": metadata["vietnamese_meaning"],
                                "category": metadata["category"],
                                "part_of_speech": metadata["part_of_speech"],
                                "example_sentences": metadata[
                                    "example_sentences"
                                ].split("|"),
                                "mnemonic_tip": metadata["mnemonic_tip"],
                                "phonetic": metadata["phonetic"],
                                "synonyms": (
                                    metadata["synonyms"].split(",")
                                    if metadata["synonyms"]
                                    else []
                                ),
                                "difficulty_level": metadata.get(
                                    "difficulty_level", "intermediate"
                                ),
                            }
                        )

                        if len(results) >= limit:
                            break

            return results

        except Exception as e:
            print(f"Error searching by category: {e}")
            return []

    def semantic_search(self, query, limit=10, similarity_threshold=0.3):
        """Semantic search with enhanced query processing using LangChain"""
        try:
            enhanced_query = self._enhance_search_query(query)

            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=limit,
                include=["metadatas", "distances", "documents"],
            )

            formatted_results = []
            if results["metadatas"] and results["metadatas"][0]:
                for i, metadata in enumerate(results["metadatas"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity = 1 - distance

                    if similarity >= similarity_threshold:
                        formatted_results.append(
                            {
                                "word": metadata["word"],
                                "vietnamese_meaning": metadata["vietnamese_meaning"],
                                "category": metadata["category"],
                                "part_of_speech": metadata["part_of_speech"],
                                "example_sentences": metadata[
                                    "example_sentences"
                                ].split("|"),
                                "mnemonic_tip": metadata["mnemonic_tip"],
                                "phonetic": metadata["phonetic"],
                                "synonyms": (
                                    metadata["synonyms"].split(",")
                                    if metadata["synonyms"]
                                    else []
                                ),
                                "difficulty_level": metadata.get(
                                    "difficulty_level", "intermediate"
                                ),
                                "similarity_score": round(similarity, 3),
                            }
                        )

            formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            print(
                f"Semantic search for '{query}' found {len(formatted_results)} results"
            )
            return formatted_results

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def _enhance_search_query(self, query):
        """Enhanced search query using LangChain for better semantic understanding"""
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a query enhancement expert. "
                        "Given a search query in Vietnamese or English, expand it with related terms "
                        "that would help find relevant vocabulary words. "
                        "Focus on synonyms, related concepts, and contextual terms. "
                        "Return the enhanced query as a single line of text.",
                    ),
                    (
                        "user",
                        "Enhance this search query for vocabulary search: {query}",
                    ),
                ]
            )

            chain = prompt | llm
            response = chain.invoke({"query": query})

            enhanced = response.content.strip()
            return f"{query} {enhanced}"

        except Exception as e:
            print(f"Error enhancing query: {e}")
            # Fallback to original method
            query_mappings = {
                "du lịch": "travel vacation holiday trip journey tourism sightseeing adventure",
                "công nghệ": "technology computer software programming internet digital tech innovation",
                "ăn uống": "food eating drinking restaurant cooking meal cuisine nutrition",
                "kinh doanh": "business work office company management finance economy",
                "giáo dục": "education school learning study teaching knowledge academic",
                "sức khỏe": "health medical doctor hospital medicine fitness wellness",
                "thể thao": "sports exercise fitness game competition athletic physical",
                "giải trí": "entertainment movie music fun leisure recreation hobby",
                "khoa học": "science research experiment discovery scientific knowledge",
                "nghệ thuật": "art creative painting drawing design artistic culture",
            }

            query_lower = query.lower()
            for vietnamese_term, english_expansion in query_mappings.items():
                if vietnamese_term in query_lower:
                    return f"{query} {english_expansion}"

            return f"{query} vocabulary words language learning"

    def delete_vocabulary(self, word):
        """Delete vocabulary from ChromaDB"""
        try:
            all_data = self.collection.get(include=["metadatas"])
            ids_to_delete = []

            if all_data["metadatas"]:
                for i, metadata in enumerate(all_data["metadatas"]):
                    if metadata.get("word", "").lower() == word.lower():
                        if i < len(all_data["ids"]):
                            ids_to_delete.append(all_data["ids"][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                print(
                    f"Deleted {len(ids_to_delete)} entries for word '{word}' from ChromaDB"
                )
                return len(ids_to_delete)
            else:
                print(f"Word '{word}' not found in ChromaDB")
                return 0

        except Exception as e:
            print(f"Error deleting vocabulary from ChromaDB: {e}")
            return 0

    def word_exists(self, word):
        """Check if vocabulary word exists in ChromaDB"""
        try:
            all_data = self.collection.get(include=["metadatas"])

            if all_data["metadatas"]:
                for metadata in all_data["metadatas"]:
                    if metadata.get("word", "").lower() == word.lower():
                        return True
            return False

        except Exception as e:
            print(f"Error checking word existence: {e}")
            return False

    def clear_all_data(self):
        """Clear all data from ChromaDB"""
        try:
            all_data = self.collection.get(include=["metadatas"])

            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
                deleted_count = len(all_data["ids"])
                print(f"Deleted {deleted_count} entries from ChromaDB")
                return deleted_count
            else:
                print("No data found in ChromaDB to delete")
                return 0

        except Exception as e:
            print(f"Error clearing ChromaDB: {e}")
            return 0

    def get_categories_stats(self):
        """Get statistics of categories"""
        try:
            all_data = self.collection.get(include=["metadatas"])
            categories = {}

            if all_data["metadatas"]:
                for metadata in all_data["metadatas"]:
                    category = metadata.get("category", "General")
                    categories[category] = categories.get(category, 0) + 1

            return categories

        except Exception as e:
            print(f"Error getting categories stats: {e}")
            return {}

    def _fallback_keyword_search(self, query, limit=10):
        """
        Fallback keyword-based search in ChromaDB when semantic search fails
        """
        try:
            # Get all data from ChromaDB
            all_data = self.collection.get(include=["metadatas", "documents"])

            if not all_data["metadatas"]:
                return []

            query_lower = query.lower()
            query_words = query_lower.split()

            results = []
            for i, metadata in enumerate(all_data["metadatas"]):
                # Search in various fields
                searchable_text = (
                    f"{metadata.get('word', '')} "
                    f"{metadata.get('vietnamese_meaning', '')} "
                    f"{metadata.get('category', '')} "
                    f"{metadata.get('example_sentences', '').replace('|', ' ')} "
                    f"{metadata.get('synonyms', '').replace(',', ' ')}"
                ).lower()

                # Calculate relevance score based on keyword matches
                relevance_score = 0
                for word in query_words:
                    if word in searchable_text:
                        relevance_score += searchable_text.count(word)

                if relevance_score > 0:
                    result_item = {
                        "word": metadata["word"],
                        "vietnamese_meaning": metadata["vietnamese_meaning"],
                        "category": metadata["category"],
                        "part_of_speech": metadata["part_of_speech"],
                        "example_sentences": metadata["example_sentences"].split("|"),
                        "mnemonic_tip": metadata["mnemonic_tip"],
                        "phonetic": metadata["phonetic"],
                        "synonyms": (
                            metadata["synonyms"].split(",")
                            if metadata["synonyms"]
                            else []
                        ),
                        "difficulty_level": metadata.get(
                            "difficulty_level", "intermediate"
                        ),
                        "similarity_score": min(
                            relevance_score / 10, 1.0
                        ),  # Normalize score
                        "source": "chromadb_keyword",
                    }
                    results.append(result_item)

            # Sort by relevance score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)

            return results[:limit]

        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []


# Initialize Vocabulary Manager
vocab_manager = VocabularyManager()

# Initialize OpenAI client (keeping for backward compatibility)
client = openai.OpenAI(
    base_url=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
)

HISTORY_FILE = "history.json"


# LangChain Tools
@tool
def semantic_search_vocabulary_tool(
    query: str, limit: int = 10, similarity_threshold: float = 0.3
) -> dict:
    """Search for vocabulary words using semantic search based on meaning and context"""
    try:
        results = vocab_manager.semantic_search(
            query, limit=limit, similarity_threshold=similarity_threshold
        )

        if not results:
            return {
                "success": False,
                "message": f"Không tìm thấy từ vựng nào liên quan đến '{query}'",
                "query": query,
                "results": [],
            }

        formatted_results = []
        for word_data in results:
            formatted_results.append(
                {
                    "word": word_data["word"],
                    "vietnamese_meaning": word_data["vietnamese_meaning"],
                    "part_of_speech": word_data["part_of_speech"],
                    "phonetic": word_data["phonetic"],
                    "example_sentences": word_data["example_sentences"],
                    "mnemonic_tip": word_data["mnemonic_tip"],
                    "difficulty_level": word_data["difficulty_level"],
                    "synonyms": word_data["synonyms"],
                    "category": word_data["category"],
                    "similarity_score": word_data["similarity_score"],
                }
            )

        return {
            "success": True,
            "message": f"Đã tìm thấy {len(results)} từ vựng liên quan đến '{query}'",
            "query": query,
            "results": formatted_results,
            "count": len(results),
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi tìm kiếm: {str(e)}",
            "query": query,
            "results": [],
        }


@tool
def add_single_vocabulary_tool(word: str) -> dict:
    """Add a single English word to the vocabulary database with detailed analysis"""
    try:
        word = word.strip()
        if not word:
            return {"success": False, "message": "Vui lòng cung cấp từ cần thêm"}

        # Check if word already exists
        chromadb_exists = vocab_manager.word_exists(word)
        history_exists = word_exists_in_history(word)

        if chromadb_exists and history_exists:
            return {
                "success": False,
                "message": f"⚠️ Từ '{word}' đã tồn tại trong flashcard!",
                "duplicate": True,
            }

        # Analyze the word
        result_data = explain_word(word)
        if not result_data["structured"]:
            return {"success": False, "message": f"❌ Không thể phân tích từ '{word}'"}

        # Save to history and database
        saved = save_to_history(word, result_data)
        if saved:
            return {
                "success": True,
                "message": f"✅ Đã thêm từ '{word}' vào flashcard!",
                "word": word,
                "formatted_result": result_data["formatted"],
                "structured_data": result_data["structured"],
                "audio_path": result_data.get("audio_path"),
            }
        else:
            return {
                "success": False,
                "message": f"⚠️ Từ '{word}' đã tồn tại trong flashcard!",
                "duplicate": True,
            }

    except Exception as e:
        return {"success": False, "message": f"Lỗi thêm từ vựng: {str(e)}"}


@tool
def analyze_text_vocabulary_tool(text: str) -> dict:
    """Extract and add important vocabulary words from a text passage"""
    try:
        if not text.strip():
            return {
                "success": False,
                "message": "Vui lòng cung cấp văn bản cần phân tích",
            }

        vocabulary_results = analyze_text_for_vocabulary(text)
        if not vocabulary_results:
            return {
                "success": False,
                "message": "❌ Không tìm thấy từ vựng quan trọng trong văn bản",
            }

        # Save vocabulary words
        added_words = []
        for vocab in vocabulary_results:
            if save_to_history(vocab["word"], vocab):
                added_words.append(vocab["word"])

        return {
            "success": True,
            "message": f"✅ Đã thêm {len(added_words)} từ vào flashcard: {', '.join(added_words)}",
            "results": vocabulary_results,
            "added_words": added_words,
            "count": len(added_words),
        }

    except Exception as e:
        return {"success": False, "message": f"Lỗi phân tích văn bản: {str(e)}"}


@tool
def get_vocabulary_stats_tool() -> dict:
    """Get statistics about the vocabulary database"""
    try:
        categories = vocab_manager.get_categories_stats()
        history = get_history()

        return {
            "success": True,
            "message": f"Thống kê từ vựng hiện tại",
            "total_words": len(history),
            "categories": categories,
            "total_categories": len(categories),
        }
    except Exception as e:
        return {"success": False, "message": f"Lỗi lấy thống kê: {str(e)}"}


# Enhanced LangChain Agent for intelligent chat
def create_intelligent_vocabulary_agent():
    """Create enhanced LangChain agent that can handle multiple types of user requests"""
    system_prompt = """Bạn là một trợ lý học từ vựng tiếng Anh thông minh và thân thiện.

    Bạn có thể giúp người dùng trong các tình huống sau:

    1. **Tìm kiếm từ vựng theo chủ đề**: Khi người dùng hỏi về từ vựng liên quan đến chủ đề nào đó
       (ví dụ: "từ vựng về du lịch", "các từ liên quan đến công nghệ", "words about emotions")
       → Sử dụng tool: semantic_search_vocabulary_tool
       → Chỉ nói ngắn gọn: "Đã tìm thấy X từ vựng về chủ đề Y" và để hệ thống hiển thị flashcard

    2. **Thêm từ đơn lẻ**: Khi người dùng muốn thêm một từ cụ thể vào flashcard
       (ví dụ: "thêm từ happy", "add word computer", "giúp tôi thêm từ beautiful")
       → Sử dụng tool: add_single_vocabulary_tool
       → Chỉ nói ngắn gọn: "Đã thêm từ X vào flashcard" và để hệ thống hiển thị flashcard

    3. **Phân tích văn bản**: Khi người dùng cung cấp một đoạn văn bản và muốn trích xuất từ vựng
       → Sử dụng tool: analyze_text_vocabulary_tool
       → Chỉ nói ngắn gọn về kết quả và để hệ thống hiển thị flashcard

    4. **Thống kê từ vựng**: Khi người dùng hỏi về số lượng từ, categories, thống kê
       → Sử dụng tool: get_vocabulary_stats_tool

    5. **Trò chuyện thông thường**: Khi người dùng chào hỏi, hỏi về cách sử dụng, hoặc các câu hỏi chung
       → Trả lời trực tiếp, thân thiện và hữu ích

    QUAN TRỌNG:
    - Luôn trả lời bằng tiếng Việt và ngắn gọn
    - Khi sử dụng tools để tìm/thêm từ vựng, KHÔNG viết danh sách chi tiết
    - Chỉ nói kết quả tóm tắt và để giao diện hiển thị flashcard đẹp
    - Luôn thân thiện và khuyến khích người dùng học tập
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    tools = [
        semantic_search_vocabulary_tool,
        add_single_vocabulary_tool,
        analyze_text_vocabulary_tool,
        get_vocabulary_stats_tool,
    ]

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
    )

    return agent_executor


# Create the enhanced agent
vocab_agent = create_intelligent_vocabulary_agent()


# Enhanced functions using LangChain
def explain_word(word):
    """Analyze vocabulary word using LangChain with structured output"""
    try:
        parser = PydanticOutputParser(pydantic_object=VocabularyWord)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful English vocabulary tutor. "
                    "Analyze the given English word thoroughly and provide structured information. "
                    "Make sure to provide accurate phonetic transcription, appropriate difficulty level, "
                    "and helpful synonyms when available.\n"
                    "{format_instructions}",
                ),
                ("user", "Please analyze the English word: '{word}'"),
            ]
        )

        chain = prompt | llm | parser

        structured_data = chain.invoke(
            {"word": word, "format_instructions": parser.get_format_instructions()}
        )

        # Convert to dict for compatibility
        structured_dict = structured_data.dict()
        formatted_result = format_vocabulary_result(structured_dict)

        return {"formatted": formatted_result, "structured": structured_dict}

    except Exception as e:
        print(f"Error in explain_word: {e}")
        # Fallback to original OpenAI method
        return explain_word_fallback(word)


def explain_word_fallback(word):
    """Fallback method using original OpenAI approach"""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful English vocabulary tutor. "
                    "When given an English word, analyze it thoroughly and use the analyze_vocabulary function "
                    "to provide structured information including Vietnamese meaning, examples, and learning tips."
                ),
            },
            {"role": "user", "content": f"Please analyze the English word: '{word}'"},
        ]

        response = client.chat.completions.create(
            model="GPT-4o-mini",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_vocabulary",
                        "description": "Analyze an English word and provide Vietnamese meaning, examples, and learning tips",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string"},
                                "vietnamese_meaning": {"type": "string"},
                                "part_of_speech": {"type": "string"},
                                "phonetic": {"type": "string"},
                                "example_sentences": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                                "mnemonic_tip": {"type": "string"},
                                "difficulty_level": {
                                    "type": "string",
                                    "enum": ["beginner", "intermediate", "advanced"],
                                },
                                "synonyms": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 3,
                                },
                            },
                            "required": [
                                "word",
                                "vietnamese_meaning",
                                "part_of_speech",
                                "example_sentences",
                                "mnemonic_tip",
                            ],
                        },
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": "analyze_vocabulary"},
            },
        )

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "analyze_vocabulary":
                function_args = json.loads(tool_call.function.arguments)
                formatted_result = format_vocabulary_result(function_args)
                return {"formatted": formatted_result, "structured": function_args}

        return {
            "formatted": f"Không thể phân tích từ '{word}'. Vui lòng thử lại.",
            "structured": None,
        }

    except Exception as e:
        print(f"Error in fallback explain_word: {e}")
        return {
            "formatted": f"Đã xảy ra lỗi khi phân tích từ '{word}': {str(e)}",
            "structured": None,
        }


def analyze_text_for_vocabulary(text):
    """Analyze text for vocabulary using LangChain with structured output"""
    try:
        parser = PydanticOutputParser(pydantic_object=VocabularyList)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an English vocabulary tutor. "
                    "Extract 5-10 important vocabulary words from the given text passage. "
                    "Focus on words that are useful for English learners - intermediate to advanced level words, "
                    "excluding very basic words like 'the', 'and', 'is', etc.\n"
                    "{format_instructions}",
                ),
                (
                    "user",
                    "Please extract important vocabulary words from this text and analyze each one:\n\n{text}",
                ),
            ]
        )

        chain = prompt | llm | parser

        result = chain.invoke(
            {"text": text, "format_instructions": parser.get_format_instructions()}
        )

        results = []
        for vocab_data in result.vocabulary_list:
            vocab_dict = vocab_data.dict()
            formatted_result = format_vocabulary_result(vocab_dict)
            results.append(
                {
                    "word": vocab_dict.get("word", ""),
                    "formatted": formatted_result,
                    "structured": vocab_dict,
                }
            )

        return results

    except Exception as e:
        print(f"Error in analyze_text_for_vocabulary: {e}")
        return []


# Rest of the utility functions remain the same
def format_vocabulary_result(function_data):
    """Format the structured data from function calling into readable text"""
    word = function_data.get("word", "")
    vietnamese_meaning = function_data.get("vietnamese_meaning", "")
    part_of_speech = function_data.get("part_of_speech", "")
    phonetic = function_data.get("phonetic", "")
    example_sentences = function_data.get("example_sentences", [])
    mnemonic_tip = function_data.get("mnemonic_tip", "")
    difficulty_level = function_data.get("difficulty_level", "")
    synonyms = function_data.get("synonyms", [])

    result = f"📚 **{word.upper()}**"

    if phonetic:
        result += f" /{phonetic}/"

    if part_of_speech:
        result += f" ({part_of_speech})"

    if difficulty_level:
        level_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
        result += f" {level_emoji.get(difficulty_level, '')} {difficulty_level.title()}"

    result += f"\n\n🇻🇳 **Nghĩa:** {vietnamese_meaning}\n\n"

    if example_sentences:
        result += "📝 **Ví dụ:**\n"
        for i, sentence in enumerate(example_sentences, 1):
            result += f"{i}. {sentence}\n"
        result += "\n"

    if synonyms:
        result += f"🔄 **Từ đồng nghĩa:** {', '.join(synonyms)}\n\n"

    result += f"💡 **Mẹo học dễ nhớ:**\n{mnemonic_tip}"

    return result


def save_to_history(word, result):
    """Save vocabulary to history with duplicate prevention"""
    chromadb_exists = vocab_manager.word_exists(word)
    history_exists = word_exists_in_history(word)

    if chromadb_exists and history_exists:
        print(f"Word '{word}' already exists in both ChromaDB and history. Skipping...")
        return False

    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                history_data = json.load(f)
            except json.JSONDecodeError:
                pass

    if result.get("structured") and result["structured"].get("vietnamese_meaning"):
        vietnamese_meaning = result["structured"]["vietnamese_meaning"]
        audio_path = tts_service.generate_audio(word, vietnamese_meaning)
        if audio_path:
            result["audio_path"] = audio_path
            result["audio_text"] = f"{word}. {vietnamese_meaning}."

        if not chromadb_exists:
            try:
                category = vocab_manager.add_vocabulary(result["structured"])
                if category:
                    result["category"] = category
                    print(f"Word '{word}' classified into category: {category}")
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")

    if not history_exists:
        history_data.append({"word": word, "result": result})
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        print(f"Word '{word}' added to history.json")

    return True


def get_history():
    """Get vocabulary history"""
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass
    return history


def word_exists_in_history(word):
    """Check if vocabulary word exists in history"""
    history = get_history()
    for item in history:
        if item.get("word", "").lower() == word.lower():
            return True
    return False


def clear_all_data():
    """Clear all data from ChromaDB and history.json"""
    try:
        chromadb_deleted = vocab_manager.clear_all_data()

        history_deleted = 0
        if os.path.exists(HISTORY_FILE):
            history_data = get_history()
            history_deleted = len(history_data)

            for item in history_data:
                if "result" in item and "audio_path" in item["result"]:
                    audio_path = item["result"]["audio_path"]
                    file_path = audio_path.replace("/static/", "static/")
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            print(f"Deleted audio file: {file_path}")
                        except Exception as e:
                            print(f"Error deleting audio file: {e}")

            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

        print(
            f"Cleared all data: {chromadb_deleted} from ChromaDB, {history_deleted} from history"
        )
        return chromadb_deleted, history_deleted

    except Exception as e:
        print(f"Error clearing all data: {e}")
        return 0, 0


def delete_flashcard(index):
    """Delete flashcard by index"""
    history_data = get_history()
    if 0 <= index < len(history_data):
        deleted_item = history_data[index]
        word_to_delete = deleted_item.get("word", "")

        if word_to_delete:
            try:
                deleted_count = vocab_manager.delete_vocabulary(word_to_delete)
                print(
                    f"Deleted {deleted_count} entries for '{word_to_delete}' from ChromaDB"
                )
            except Exception as e:
                print(f"Error deleting from ChromaDB: {e}")

        if "result" in deleted_item and "audio_path" in deleted_item["result"]:
            audio_path = deleted_item["result"]["audio_path"]
            file_path = audio_path.replace("/static/", "static/")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted audio file: {file_path}")
                except Exception as e:
                    print(f"Error deleting audio file: {e}")

        del history_data[index]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)


def extract_word_from_input(user_input):
    """Extract the actual English word from user input phrases"""
    user_input = user_input.lower().strip()

    english_phrases = [
        "help me add world",
        "help me add word",
        "add world",
        "add word",
        "please add",
        "can you add",
        "i want to add",
        "add the word",
        "add this word",
        "add vocabulary",
        "help me add",
        "please help me add",
        "can you help me add",
        "i need to add",
        "i would like to add",
        "add the vocabulary",
        "add this vocabulary",
        "add new word",
        "add new vocabulary",
    ]

    vietnamese_phrases = [
        "giúp tôi thêm từ",
        "giúp tôi thêm",
        "thêm từ",
        "thêm từ vựng",
        "tôi muốn thêm từ",
        "hãy thêm từ",
        "thêm từ này",
        "thêm từ đó",
        "thêm từ tiếng anh",
        "thêm từ mới",
        "giúp tôi thêm từ vựng",
        "tôi cần thêm từ",
        "bạn có thể thêm từ",
        "hãy giúp tôi thêm từ",
        "thêm từ vựng mới",
        "thêm từ này vào",
        "thêm từ đó vào",
    ]

    for phrase in english_phrases + vietnamese_phrases:
        user_input = user_input.replace(phrase, "").strip()

    user_input = re.sub(r"\s+", " ", user_input).strip()
    user_input = re.sub(r"[^\w\s]", "", user_input).strip()

    words = user_input.split()

    if len(words) == 1:
        return words[0]
    elif len(words) > 1:
        filler_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "me",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "hers",
            "ours",
            "theirs",
        }

        meaningful_words = [word for word in words if word.lower() not in filler_words]

        if meaningful_words:
            meaningful_words.sort(key=len, reverse=True)
            return meaningful_words[0]
        else:
            return words[-1]

    return user_input


# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    word = None
    if request.method == "POST":
        word = request.form["word"].strip()
        if word:
            result_data = explain_word(word)
            result = result_data["formatted"]
            save_to_history(word, result_data)

    history = get_history()
    history = history[::-1]

    return render_template("index.html", result=result, word=word, history=history)


@app.route("/chat")
def chat():
    """Chat page"""
    history = get_history()
    return render_template("chat.html", total_words=len(history))


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """Enhanced API for intelligent chat processing using LangChain agent"""
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"success": False, "message": "Vui lòng nhập nội dung"})

    try:
        # Use the intelligent vocabulary agent to process the message
        response = vocab_agent.invoke({"input": message})

        # Extract the response content
        if isinstance(response, dict):
            agent_response = response.get("output", str(response))
            intermediate_steps = response.get("intermediate_steps", [])
        else:
            agent_response = str(response)
            intermediate_steps = []

        # Parse tool results from intermediate steps
        tool_results = extract_tool_results_from_agent_steps(
            intermediate_steps, message
        )

        # Check if this looks like a vocabulary search result and enhance the display
        enhanced_response = enhance_vocabulary_response(agent_response, tool_results)

        return jsonify(
            {
                "success": True,
                "message": enhanced_response["message"],
                "type": "intelligent_chat",
                "tool_results": enhanced_response["tool_results"],
                "vocabulary_cards": enhanced_response.get("vocabulary_cards", []),
                "has_structured_data": len(
                    enhanced_response.get("vocabulary_cards", [])
                )
                > 0,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        print(f"Error in intelligent chat: {e}")
        return jsonify(
            {
                "success": False,
                "message": f"❌ Đã xảy ra lỗi: {str(e)}",
                "error_type": "agent_error",
            }
        )


def extract_tool_results_from_agent_steps(intermediate_steps, original_message):
    """Extract structured results from agent's tool usage"""
    tool_results = {
        "vocabulary_results": [],
        "search_results": [],
        "stats": {},
        "added_words": [],
    }

    for step in intermediate_steps:
        if len(step) >= 2:
            action, observation = step[0], step[1]

            if hasattr(action, "tool") and hasattr(action, "tool_input"):
                tool_name = action.tool
                tool_input = action.tool_input

                # Handle different tool results
                if tool_name == "semantic_search_vocabulary_tool" and observation:
                    try:
                        if isinstance(observation, dict) and observation.get("success"):
                            tool_results["search_results"].extend(
                                observation.get("results", [])
                            )
                    except Exception as e:
                        print(f"Error parsing search results: {e}")

                elif tool_name == "add_single_vocabulary_tool" and observation:
                    try:
                        if isinstance(observation, dict) and observation.get("success"):
                            if observation.get("structured_data"):
                                tool_results["vocabulary_results"].append(
                                    {
                                        "structured": observation["structured_data"],
                                        "word": observation.get("word", ""),
                                        "formatted": observation.get(
                                            "formatted_result", ""
                                        ),
                                    }
                                )
                            tool_results["added_words"].append(
                                observation.get("word", "")
                            )
                    except Exception as e:
                        print(f"Error parsing add word results: {e}")

                elif tool_name == "analyze_text_vocabulary_tool" and observation:
                    try:
                        if isinstance(observation, dict) and observation.get("success"):
                            tool_results["vocabulary_results"].extend(
                                observation.get("results", [])
                            )
                            tool_results["added_words"].extend(
                                observation.get("added_words", [])
                            )
                    except Exception as e:
                        print(f"Error parsing text analysis results: {e}")

                elif tool_name == "get_vocabulary_stats_tool" and observation:
                    try:
                        if isinstance(observation, dict) and observation.get("success"):
                            tool_results["stats"] = {
                                "total_words": observation.get("total_words", 0),
                                "categories": observation.get("categories", {}),
                                "total_categories": observation.get(
                                    "total_categories", 0
                                ),
                            }
                    except Exception as e:
                        print(f"Error parsing stats results: {e}")

    return tool_results


def enhance_vocabulary_response(agent_response, tool_results):
    """Enhance the response with structured vocabulary data for beautiful display"""
    vocabulary_cards = []
    enhanced_message = agent_response

    # If we have search results, create vocabulary cards
    if tool_results.get("search_results"):
        vocabulary_cards = []
        for result in tool_results["search_results"]:
            card_data = {
                "word": result.get("word", ""),
                "vietnamese_meaning": result.get("vietnamese_meaning", ""),
                "part_of_speech": result.get("part_of_speech", ""),
                "phonetic": result.get("phonetic", ""),
                "example_sentences": result.get("example_sentences", []),
                "mnemonic_tip": result.get("mnemonic_tip", ""),
                "difficulty_level": result.get("difficulty_level", "intermediate"),
                "synonyms": result.get("synonyms", []),
                "category": result.get("category", "General"),
                "similarity_score": result.get("similarity_score", 0),
                "audio_path": None,  # Will be generated on demand
            }
            vocabulary_cards.append(card_data)

        # Create a clean header message for search results
        if len(vocabulary_cards) > 0:
            enhanced_message = f"🎯 Tìm thấy {len(vocabulary_cards)} từ vựng phù hợp! Dưới đây là các flashcard chi tiết:"

    # If we have added words, create vocabulary cards
    elif tool_results.get("vocabulary_results"):
        for result in tool_results["vocabulary_results"]:
            if result.get("structured"):
                card_data = result["structured"].copy()
                card_data["audio_path"] = None  # Will be generated on demand
                vocabulary_cards.append(card_data)

        if len(vocabulary_cards) > 0:
            words_list = ", ".join([card["word"] for card in vocabulary_cards])
            enhanced_message = (
                f"✅ Đã thêm {len(vocabulary_cards)} từ vào flashcard: {words_list}"
            )

    return {
        "message": enhanced_message,
        "tool_results": tool_results,
        "vocabulary_cards": vocabulary_cards,
    }


@app.route("/api/chat-legacy", methods=["POST"])
def chat_api_legacy():
    """Legacy API for processing chat messages with explicit types (kept for backward compatibility)"""
    data = request.get_json()
    message = data.get("message", "").strip()
    chat_type = data.get("type", "")

    if not message:
        return jsonify({"success": False, "message": "Vui lòng nhập nội dung"})

    try:
        if chat_type == "word":
            word_to_analyze = extract_word_from_input(message)
            print(f"User input: '{message}' -> Extracted word: '{word_to_analyze}'")

            result_data = explain_word(word_to_analyze)
            if result_data["structured"]:
                chromadb_exists = vocab_manager.word_exists(word_to_analyze)
                history_exists = word_exists_in_history(word_to_analyze)

                if chromadb_exists and history_exists:
                    return jsonify(
                        {
                            "success": False,
                            "message": f"⚠️ Từ '{word_to_analyze}' đã tồn tại trong flashcard!",
                            "result": result_data["formatted"],
                            "structured_data": result_data["structured"],
                            "type": "word",
                            "duplicate": True,
                        }
                    )

                saved = save_to_history(word_to_analyze, result_data)
                if saved:
                    return jsonify(
                        {
                            "success": True,
                            "message": f"✅ Đã thêm từ '{word_to_analyze}' vào flashcard!",
                            "result": result_data["formatted"],
                            "structured_data": result_data["structured"],
                            "audio_path": result_data.get("audio_path"),
                            "type": "word",
                            "extracted_word": word_to_analyze,
                            "original_input": message,
                        }
                    )
                else:
                    return jsonify(
                        {
                            "success": False,
                            "message": f"⚠️ Từ '{word_to_analyze}' đã tồn tại trong flashcard!",
                            "result": result_data["formatted"],
                            "structured_data": result_data["structured"],
                            "type": "word",
                            "duplicate": True,
                            "extracted_word": word_to_analyze,
                            "original_input": message,
                        }
                    )
            else:
                return jsonify(
                    {
                        "success": False,
                        "message": f"❌ Không thể phân tích từ '{word_to_analyze}'",
                        "extracted_word": word_to_analyze,
                        "original_input": message,
                    }
                )

        elif chat_type == "text":
            vocabulary_results = analyze_text_for_vocabulary(message)
            if vocabulary_results:
                for vocab in vocabulary_results:
                    save_to_history(vocab["word"], vocab)

                words_added = [vocab["word"] for vocab in vocabulary_results]
                return jsonify(
                    {
                        "success": True,
                        "message": f"✅ Đã thêm {len(vocabulary_results)} từ vào flashcard: {', '.join(words_added)}",
                        "results": vocabulary_results,
                        "type": "text",
                        "count": len(vocabulary_results),
                    }
                )
            else:
                return jsonify(
                    {
                        "success": False,
                        "message": "❌ Không tìm thấy từ vựng quan trọng trong văn bản",
                    }
                )
        else:
            return jsonify(
                {"success": False, "message": "❌ Loại yêu cầu không hợp lệ"}
            )

    except Exception as e:
        return jsonify({"success": False, "message": f"❌ Đã xảy ra lỗi: {str(e)}"})


@app.route("/api/semantic-search", methods=["POST"])
def semantic_search_api():
    """Enhanced API for semantic search using ChromaDB first, then OpenAI fallback"""
    data = request.get_json()
    message = data.get("message", "").strip()
    auto_add = data.get("auto_add", False)  # New option to auto-add OpenAI results

    if not message:
        return jsonify({"success": False, "message": "Vui lòng nhập nội dung tìm kiếm"})

    try:
        # Step 1: Try ChromaDB semantic search first
        chromadb_results = vocab_manager.semantic_search(
            message, limit=10, similarity_threshold=0.3
        )

        if chromadb_results:
            print(f"Found {len(chromadb_results)} results in ChromaDB")

            # Format results for response
            structured_results = []
            for vocab in chromadb_results:
                structured_results.append(
                    {
                        "word": vocab["word"],
                        "structured": vocab,
                        "formatted": format_vocabulary_result(vocab),
                    }
                )

            return jsonify(
                {
                    "success": True,
                    "source": "chromadb",
                    "message": f"🔍 Tìm thấy {len(chromadb_results)} từ vựng liên quan đến '{message}' trong cơ sở dữ liệu",
                    "type": "semantic_search",
                    "query": message,
                    "results": chromadb_results,
                    "count": len(chromadb_results),
                    "structured_results": structured_results,
                }
            )

        # Step 2: If no results in ChromaDB, use OpenAI
        print(f"No results in ChromaDB, using OpenAI for: '{message}'")

        # Use LangChain to generate relevant vocabulary
        try:
            from langchain_core.output_parsers import PydanticOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            # Define the search result structure for OpenAI
            class OpenAIVocabularySearch(BaseModel):
                words: List[VocabularyWord] = Field(
                    description="List of vocabulary words related to the search query",
                    min_items=3,
                    max_items=10,
                )

            parser = PydanticOutputParser(pydantic_object=OpenAIVocabularySearch)

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an English vocabulary expert for Vietnamese learners. "
                        "Given a search query (in Vietnamese or English), generate 5-8 relevant "
                        "English vocabulary words that would be most useful for learners studying this topic. "
                        "Focus on practical, commonly used words. Provide accurate Vietnamese meanings, "
                        "clear example sentences, and helpful learning tips.\n"
                        "{format_instructions}",
                    ),
                    (
                        "user",
                        "Generate English vocabulary words related to: '{query}'\n"
                        "Consider the most important and useful words that English learners "
                        "should know about this topic.",
                    ),
                ]
            )

            chain = prompt | llm | parser

            result = chain.invoke(
                {
                    "query": message,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            # Convert to expected format
            openai_results = []
            structured_results = []

            for word_data in result.words:
                word_dict = word_data.dict()

                # Add metadata
                word_dict["similarity_score"] = (
                    0.95  # High score for AI-generated relevant words
                )
                word_dict["source"] = "openai_generated"
                word_dict["category"] = vocab_manager.classify_category(
                    word_dict["word"],
                    word_dict["vietnamese_meaning"],
                    word_dict["part_of_speech"],
                    word_dict["example_sentences"],
                )

                openai_results.append(word_dict)

                # Format for display
                structured_results.append(
                    {
                        "word": word_dict["word"],
                        "structured": word_dict,
                        "formatted": format_vocabulary_result(word_dict),
                    }
                )

            # Optional: Auto-add to database if requested
            added_words = []
            if auto_add and openai_results:
                for word_data in openai_results:
                    try:
                        # Check if word already exists
                        if not vocab_manager.word_exists(word_data["word"]):
                            # Add to ChromaDB
                            vocab_manager.add_vocabulary(word_data)

                            # Add to history with audio generation
                            result_data = {
                                "formatted": format_vocabulary_result(word_data),
                                "structured": word_data,
                            }

                            if save_to_history(word_data["word"], result_data):
                                added_words.append(word_data["word"])

                    except Exception as e:
                        print(f"Error adding word {word_data['word']}: {e}")

            response_message = f"🤖 Tìm thấy {len(openai_results)} từ vựng liên quan đến '{message}' (được tạo bởi AI)"
            if added_words:
                response_message += f" và đã thêm {len(added_words)} từ vào flashcard"

            return jsonify(
                {
                    "success": True,
                    "source": "openai",
                    "message": response_message,
                    "type": "semantic_search",
                    "query": message,
                    "results": openai_results,
                    "count": len(openai_results),
                    "structured_results": structured_results,
                    "added_words": added_words if added_words else [],
                }
            )

        except Exception as openai_error:
            print(f"Error with OpenAI search: {openai_error}")

            # Step 3: Final fallback - try keyword search in ChromaDB
            print("Trying keyword fallback search...")
            keyword_results = vocab_manager._fallback_keyword_search(message, limit=10)

            if keyword_results:
                structured_results = []
                for vocab in keyword_results:
                    structured_results.append(
                        {
                            "word": vocab["word"],
                            "structured": vocab,
                            "formatted": format_vocabulary_result(vocab),
                        }
                    )

                return jsonify(
                    {
                        "success": True,
                        "source": "chromadb_keyword",
                        "message": f"🔍 Tìm thấy {len(keyword_results)} từ vựng có liên quan đến '{message}' (tìm kiếm từ khóa)",
                        "type": "semantic_search",
                        "query": message,
                        "results": keyword_results,
                        "count": len(keyword_results),
                        "structured_results": structured_results,
                    }
                )

            # No results from any method
            return jsonify(
                {
                    "success": False,
                    "source": "none",
                    "message": f"❌ Không tìm thấy từ vựng nào liên quan đến '{message}'. Hãy thử với từ khóa khác.",
                    "type": "semantic_search",
                    "query": message,
                    "results": [],
                    "count": 0,
                }
            )

    except Exception as e:
        print(f"Error in semantic_search_api: {e}")
        return jsonify(
            {"success": False, "message": f"⚠️ Đã xảy ra lỗi tìm kiếm: {str(e)}"}
        )


@app.route("/list")
def flashcard_list():
    """Display list of all flashcards"""
    history = get_history()
    history = history[::-1]
    return render_template("list.html", history=history)


@app.route("/delete/<int:index>")
def delete_flashcard_route(index):
    """Delete flashcard by index"""
    history = get_history()
    actual_index = len(history) - 1 - index
    delete_flashcard(actual_index)
    return redirect(url_for("flashcard_list"))


@app.route("/api/delete-word", methods=["POST"])
def delete_word_api():
    """API for deleting vocabulary word by name"""
    try:
        data = request.get_json()
        word = data.get("word", "").strip()

        if not word:
            return jsonify(
                {"success": False, "message": "Vui lòng cung cấp tên từ cần xóa"}
            )

        deleted_count = vocab_manager.delete_vocabulary(word)

        history_data = get_history()
        original_count = len(history_data)
        history_data = [
            item
            for item in history_data
            if item.get("word", "").lower() != word.lower()
        ]

        if len(history_data) < original_count:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)

        if deleted_count > 0 or len(history_data) < original_count:
            return jsonify(
                {
                    "success": True,
                    "message": f"Đã xóa từ '{word}' thành công",
                    "deleted_from_chromadb": deleted_count,
                    "deleted_from_history": original_count - len(history_data),
                }
            )
        else:
            return jsonify(
                {"success": False, "message": f"Không tìm thấy từ '{word}' để xóa"}
            )

    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi xóa từ vựng: {str(e)}"})


@app.route("/api/clear-all-data", methods=["POST"])
def clear_all_data_api():
    """API for clearing all data from ChromaDB and history.json"""
    try:
        data = request.get_json()
        confirm = data.get("confirm", False)

        if not confirm:
            return jsonify(
                {
                    "success": False,
                    "message": "Vui lòng xác nhận bằng cách gửi 'confirm': true",
                }
            )

        chromadb_deleted, history_deleted = clear_all_data()

        return jsonify(
            {
                "success": True,
                "message": f"Đã xóa toàn bộ dữ liệu thành công",
                "deleted_from_chromadb": chromadb_deleted,
                "deleted_from_history": history_deleted,
                "total_deleted": chromadb_deleted + history_deleted,
            }
        )

    except Exception as e:
        return jsonify(
            {"success": False, "message": f"Lỗi xóa toàn bộ dữ liệu: {str(e)}"}
        )


@app.route("/api/word/<word>")
def get_word_api(word):
    """API endpoint to get word information in JSON format"""
    result_data = explain_word(word)
    return {
        "word": word,
        "formatted_result": result_data["formatted"],
        "structured_data": result_data["structured"],
        "audio_path": result_data.get("audio_path"),
    }


@app.route("/api/search-category", methods=["POST"])
def search_by_category_api():
    """API for searching vocabulary by category"""
    try:
        data = request.get_json()
        category = data.get("category", "").strip()
        limit = data.get("limit", 50)

        if not category:
            return jsonify({"success": False, "message": "Vui lòng chọn category"})

        results = vocab_manager.search_by_category(category, limit=limit)

        if results:
            return jsonify(
                {
                    "success": True,
                    "category": category,
                    "results": results,
                    "count": len(results),
                    "message": f"Tìm thấy {len(results)} từ vựng trong category '{category}'",
                }
            )
        else:
            return jsonify(
                {
                    "success": False,
                    "message": f"Không tìm thấy từ vựng nào trong category '{category}'",
                }
            )

    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi tìm kiếm: {str(e)}"})


@app.route("/api/categories")
def get_categories_api():
    """API for getting category statistics"""
    try:
        categories = vocab_manager.get_categories_stats()
        return jsonify(
            {
                "success": True,
                "categories": categories,
                "total_words": sum(categories.values()),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi lấy thống kê: {str(e)}"})


@app.route("/search")
def search_page():
    """Smart search page with semantic search"""
    return render_template("search.html")


@app.route("/api/generate-audio/<word>")
def generate_audio_api(word):
    """Generate or get existing audio for a word"""
    try:
        history = get_history()
        for item in history:
            if item["word"].lower() == word.lower():
                result = item.get("result", {})

                if result.get("audio_path"):
                    return jsonify(
                        {
                            "success": True,
                            "audio_path": result["audio_path"],
                            "word": word,
                        }
                    )

                if result.get("structured") and result["structured"].get(
                    "vietnamese_meaning"
                ):
                    vietnamese_meaning = result["structured"]["vietnamese_meaning"]
                    audio_path = tts_service.generate_audio(word, vietnamese_meaning)
                    if audio_path:
                        result["audio_path"] = audio_path
                        result["audio_text"] = f"{word}. {vietnamese_meaning}."

                        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                            json.dump(history, f, ensure_ascii=False, indent=2)

                        return jsonify(
                            {"success": True, "audio_path": audio_path, "word": word}
                        )

        return jsonify(
            {"success": False, "message": "Word not found or cannot generate audio"}
        )

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route("/static/audio/<filename>")
def serve_audio(filename):
    """Serve audio files"""
    return send_from_directory("static/audio", filename)


# FAQ page
@app.route("/faq")
def faq_page():
    return render_template("faq.html")


# FAQ RAG setup
FAQ_FILE = "faq.json"
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "faq_collection"


def setup_faq_vectorstore():
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        api_version="2023-05-15",
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    )

    # Create Chroma client
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Load documents
    docs = load_faq_documents()

    # Create or get existing vectorstore
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Add documents if collection is empty
    if len(vectorstore.get()["ids"]) == 0:
        vectorstore.add_documents(docs)
        vectorstore.persist()

    return vectorstore


def load_faq_documents():
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    docs = [
        Document(
            page_content=faq["question"] + " " + faq["answer"],
            metadata={"question": faq["question"], "answer": faq["answer"]},
        )
        for faq in faq_data
    ]
    return docs


# Only initialize once
faq_vectorstore = None


def get_faq_vectorstore():
    global faq_vectorstore
    if faq_vectorstore is None:
        faq_vectorstore = setup_faq_vectorstore()
    return faq_vectorstore


def rag_faq_answer(query):
    vs = get_faq_vectorstore()
    docs = vs.similarity_search(query, k=2)

    if not docs:
        return "Sorry, I couldn't find an answer."

    # Initialize Azure OpenAI Chat model
    llm = AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
        deployment_name=os.getenv("AZURE_OPENAI_LLM_MODEL"),
        azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    )

    # Prepare context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions about the flash card system.",
            ),
            (
                "human",
                """Based on the following context, please answer the question. 
                     If you cannot find a relevant answer in the context, say so.
                     
                     Context: {context}
                     
                     Question: {question}""",
            ),
        ]
    )

    # Create and invoke chain
    chain = prompt | llm

    # Get response
    response = chain.invoke({"context": context, "question": query})

    return response.content


# FAQ API
@app.route("/api/faq", methods=["POST"])
def faq_api():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"success": False, "answer": "No question provided."})
    answer = rag_faq_answer(question)
    return jsonify({"success": True, "answer": answer})


if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True, port=6969)
