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

load_dotenv()

app = Flask(__name__)


# TTS Service Class
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
            print(f"Error loading TTS models: {e}")
            self.model = None
            self.tokenizer = None

    def generate_audio(self, word, vietnamese_meaning):
        """Generate audio for a word with Vietnamese meaning"""
        if not self.model or not self.tokenizer:
            print("TTS models not available")
            return None

        try:
            # Create text for TTS (English word + Vietnamese meaning)
            tts_text = f"{word}"

            # Generate unique filename based on content hash
            content_hash = hashlib.md5(tts_text.encode()).hexdigest()[:8]
            filename = f"{word}_{content_hash}.wav"
            filepath = os.path.join(self.audio_dir, filename)

            # Check if file already exists
            if os.path.exists(filepath):
                return f"/static/audio/{filename}"

            # Generate audio
            inputs = self.tokenizer(tts_text, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**inputs).waveform

            # Save audio file
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


# Vocabulary Management with ChromaDB
class VocabularyManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="vocabulary", metadata={"hnsw:space": "cosine"}
        )

    def classify_category(
        self, word, vietnamese_meaning, part_of_speech, example_sentences
    ):
        """Phân loại category cho từ vựng dựa trên AI"""
        try:
            # Tạo context để phân loại
            context = f"Word: {word}\nMeaning: {vietnamese_meaning}\nPart of speech: {part_of_speech}\nExamples: {' '.join(example_sentences)}"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a vocabulary categorization expert. "
                        "Classify the given English word into ONE of these categories: "
                        "Business, Technology, Education, Health, Travel, Food, Sports, "
                        "Entertainment, Science, Art, Nature, Family, Emotions, Time, "
                        "Colors, Numbers, Animals, Transportation, Clothing, Weather. "
                        "Return ONLY the category name, nothing else."
                    ),
                },
                {"role": "user", "content": f"Classify this word: {context}"},
            ]

            response = client.chat.completions.create(
                model="GPT-4o-mini",
                messages=messages,
                max_tokens=20,
                temperature=0.1,
            )

            category = response.choices[0].message.content.strip()
            return category

        except Exception as e:
            print(f"Error classifying category: {e}")
            return "General"

    def add_vocabulary(self, word_data):
        """Thêm từ vựng vào ChromaDB với category tự động"""
        try:
            word = word_data.get("word", "")
            vietnamese_meaning = word_data.get("vietnamese_meaning", "")
            part_of_speech = word_data.get("part_of_speech", "")
            example_sentences = word_data.get("example_sentences", [])

            # Phân loại category tự động
            category = self.classify_category(
                word, vietnamese_meaning, part_of_speech, example_sentences
            )

            # Tạo document text để embedding
            document_text = f"{word} {vietnamese_meaning} {part_of_speech} {' '.join(example_sentences)}"

            # Tạo metadata
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

            # Thêm vào collection
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

    # def search_by_topic(self, topic, limit=10, similarity_threshold=0.5):
    #     """Tìm kiếm từ vựng theo chủ đề - ĐÃ XÓA"""
    #     return []

    def search_by_category(self, category, limit=50):
        """Tìm kiếm từ vựng theo category cụ thể - DEPRECATED, sử dụng semantic_search thay thế"""
        try:
            # Lấy tất cả dữ liệu
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
        """Tìm kiếm từ vựng bằng semantic search dựa trên vector embeddings"""
        try:
            # Tạo query text phong phú hơn để tìm kiếm semantic
            enhanced_query = self._enhance_search_query(query)

            # Thực hiện semantic search với ChromaDB
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=limit,
                include=["metadatas", "distances", "documents"],
            )

            formatted_results = []
            if results["metadatas"] and results["metadatas"][0]:
                for i, metadata in enumerate(results["metadatas"][0]):
                    # Kiểm tra similarity threshold
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity = (
                        1 - distance
                    )  # ChromaDB trả về distance, chuyển thành similarity

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

            # Sắp xếp theo similarity score giảm dần
            formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)

            print(
                f"Semantic search for '{query}' found {len(formatted_results)} results"
            )
            return formatted_results

        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def _enhance_search_query(self, query):
        """Tăng cường query để semantic search hiệu quả hơn"""
        # Mapping các từ khóa tiếng Việt sang tiếng Anh và mở rộng ngữ cảnh
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
            "thiên nhiên": "nature environment natural outdoor wildlife plants animals",
            "gia đình": "family parents children relatives home domestic",
            "cảm xúc": "emotions feelings mood happy sad angry love",
            "thời gian": "time clock hour minute day week month year",
            "màu sắc": "colors red blue green yellow black white colorful",
            "số": "numbers counting mathematics numeric quantity amount",
            "động vật": "animals pets wildlife creatures living beings",
            "giao thông": "transportation vehicle car bus train plane travel",
            "quần áo": "clothing clothes fashion wear dress shirt pants",
            "thời tiết": "weather climate rain sun snow wind temperature",
        }

        # Tìm mapping phù hợp
        query_lower = query.lower()
        for vietnamese_term, english_expansion in query_mappings.items():
            if vietnamese_term in query_lower:
                return f"{query} {english_expansion}"

        # Nếu không tìm thấy mapping, trả về query gốc với một số từ khóa chung
        return f"{query} vocabulary words language learning"

    def delete_vocabulary(self, word):
        """Xóa từ vựng khỏi ChromaDB"""
        try:
            # Lấy tất cả dữ liệu để tìm IDs của từ cần xóa
            all_data = self.collection.get(include=["metadatas"])
            ids_to_delete = []

            if all_data["metadatas"]:
                for i, metadata in enumerate(all_data["metadatas"]):
                    if metadata.get("word", "").lower() == word.lower():
                        # Lấy ID tương ứng
                        if i < len(all_data["ids"]):
                            ids_to_delete.append(all_data["ids"][i])

            # Xóa các IDs tìm được
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
        """Kiểm tra xem từ vựng đã tồn tại trong ChromaDB chưa"""
        try:
            # Lấy tất cả metadata
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
        """Xóa toàn bộ dữ liệu trong ChromaDB"""
        try:
            # Lấy tất cả IDs
            all_data = self.collection.get(include=["metadatas"])

            if all_data["ids"]:
                # Xóa tất cả
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
        """Lấy thống kê các category"""
        try:
            # Lấy tất cả metadata
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


# Initialize Vocabulary Manager
vocab_manager = VocabularyManager()

# Khởi tạo OpenAI client
client = openai.OpenAI(
    base_url=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
)

HISTORY_FILE = "history.json"

# Function schemas remain the same
VOCABULARY_FUNCTION = {
    "type": "function",
    "function": {
        "name": "analyze_vocabulary",
        "description": "Analyze an English word and provide Vietnamese meaning, examples, and learning tips",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The English word being analyzed, fix this word if it is not a valid English word",
                },
                "vietnamese_meaning": {
                    "type": "string",
                    "description": "The meaning of the word in Vietnamese",
                },
                "part_of_speech": {
                    "type": "string",
                    "description": "Part of speech (noun, verb, adjective, etc.)",
                },
                "phonetic": {
                    "type": "string",
                    "description": "Phonetic pronunciation of the word",
                },
                "example_sentences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Two example sentences using the word",
                    "minItems": 2,
                    "maxItems": 2,
                },
                "mnemonic_tip": {
                    "type": "string",
                    "description": "A memorable tip or mnemonic to help learn the word",
                },
                "difficulty_level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "Difficulty level of the word",
                },
                "synonyms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of synonyms (up to 3)",
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

TEXT_ANALYSIS_FUNCTION = {
    "type": "function",
    "function": {
        "name": "extract_vocabulary_from_text",
        "description": "Extract important vocabulary words from a text passage and analyze each word",
        "parameters": {
            "type": "object",
            "properties": {
                "vocabulary_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "word": {
                                "type": "string",
                                "description": "The English word being analyzed",
                            },
                            "vietnamese_meaning": {
                                "type": "string",
                                "description": "The meaning of the word in Vietnamese",
                            },
                            "part_of_speech": {
                                "type": "string",
                                "description": "Part of speech (noun, verb, adjective, etc.)",
                            },
                            "phonetic": {
                                "type": "string",
                                "description": "Phonetic pronunciation of the word",
                            },
                            "example_sentences": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Two example sentences using the word",
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "mnemonic_tip": {
                                "type": "string",
                                "description": "A memorable tip or mnemonic to help learn the word",
                            },
                            "difficulty_level": {
                                "type": "string",
                                "enum": ["beginner", "intermediate", "advanced"],
                                "description": "Difficulty level of the word",
                            },
                            "synonyms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of synonyms (up to 3)",
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
                    "description": "List of vocabulary words extracted from the text",
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
            "required": ["vocabulary_list"],
        },
    },
}

# Function tool for semantic vocabulary search
SEMANTIC_VOCABULARY_SEARCH_FUNCTION = {
    "type": "function",
    "function": {
        "name": "semantic_search_vocabulary",
        "description": "Tìm kiếm từ vựng bằng semantic search dựa trên ý nghĩa và ngữ cảnh. Sử dụng khi người dùng hỏi về từ vựng liên quan đến một chủ đề, khái niệm hoặc tình huống cụ thể.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Từ khóa hoặc mô tả chủ đề cần tìm kiếm. Có thể là tiếng Việt hoặc tiếng Anh. Ví dụ: 'du lịch', 'travel', 'công nghệ', 'technology', 'ăn uống', 'food', 'cảm xúc vui buồn', 'happy sad emotions'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Số lượng từ vựng tối đa cần trả về (mặc định 10)",
                    "default": 10,
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Ngưỡng độ tương đồng tối thiểu (0.0-1.0, mặc định 0.3)",
                    "default": 0.3,
                },
            },
            "required": ["query"],
        },
    },
}


# Enhanced save_to_history function with audio and ChromaDB (with duplicate prevention)
def save_to_history(word, result):
    # Kiểm tra duplicate trước khi lưu
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

    # Generate audio if structured data is available
    if result.get("structured") and result["structured"].get("vietnamese_meaning"):
        vietnamese_meaning = result["structured"]["vietnamese_meaning"]
        audio_path = tts_service.generate_audio(word, vietnamese_meaning)
        if audio_path:
            result["audio_path"] = audio_path
            result["audio_text"] = f"{word}. {vietnamese_meaning}."

        # Thêm vào ChromaDB với phân loại tự động (chỉ khi chưa tồn tại)
        if not chromadb_exists:
            try:
                category = vocab_manager.add_vocabulary(result["structured"])
                if category:
                    result["category"] = category
                    print(f"Word '{word}' classified into category: {category}")
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")
        else:
            print(
                f"Word '{word}' already exists in ChromaDB. Skipping ChromaDB insertion..."
            )

    # Thêm vào history.json (chỉ khi chưa tồn tại)
    if not history_exists:
        history_data.append({"word": word, "result": result})
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        print(f"Word '{word}' added to history.json")
    else:
        print(
            f"Word '{word}' already exists in history.json. Skipping history insertion..."
        )

    return True


def get_history():
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass
    return history


def word_exists_in_history(word):
    """Kiểm tra xem từ vựng đã tồn tại trong history.json chưa"""
    history = get_history()
    for item in history:
        if item.get("word", "").lower() == word.lower():
            return True
    return False


def clear_all_data():
    """Xóa toàn bộ dữ liệu từ ChromaDB và history.json"""
    try:
        # Xóa ChromaDB
        chromadb_deleted = vocab_manager.clear_all_data()

        # Xóa history.json
        history_deleted = 0
        if os.path.exists(HISTORY_FILE):
            history_data = get_history()
            history_deleted = len(history_data)

            # Xóa tất cả file audio
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

            # Xóa history.json
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
    history_data = get_history()
    if 0 <= index < len(history_data):
        # Get the word to delete
        deleted_item = history_data[index]
        word_to_delete = deleted_item.get("word", "")

        # Delete from ChromaDB first
        if word_to_delete:
            try:
                deleted_count = vocab_manager.delete_vocabulary(word_to_delete)
                print(
                    f"Deleted {deleted_count} entries for '{word_to_delete}' from ChromaDB"
                )
            except Exception as e:
                print(f"Error deleting from ChromaDB: {e}")

        # Delete associated audio file if it exists
        if "result" in deleted_item and "audio_path" in deleted_item["result"]:
            audio_path = deleted_item["result"]["audio_path"]
            # Convert web path to file path
            file_path = audio_path.replace("/static/", "static/")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted audio file: {file_path}")
                except Exception as e:
                    print(f"Error deleting audio file: {e}")

        # Delete from history.json
        del history_data[index]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)


def semantic_search_vocabulary_function(query, limit=10, similarity_threshold=0.3):
    """Function để tìm kiếm từ vựng bằng semantic search cho function calling"""
    try:
        # Tìm kiếm bằng semantic search
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

        # Format kết quả cho function calling
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
            "message": f"Tìm thấy {len(results)} từ vựng liên quan đến '{query}'",
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


def explain_word(word):
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful English vocabulary tutor. "
                    "When given an English word, analyze it thoroughly and use the analyze_vocabulary function "
                    "to provide structured information including Vietnamese meaning, examples, and learning tips. "
                    "Make sure to provide accurate phonetic transcription, appropriate difficulty level, "
                    "and helpful synonyms when available."
                ),
            },
            {"role": "user", "content": f"Please analyze the English word: '{word}'"},
        ]

        response = client.chat.completions.create(
            model="GPT-4o-mini",
            messages=messages,
            tools=[VOCABULARY_FUNCTION],
            tool_choice={
                "type": "function",
                "function": {"name": "analyze_vocabulary"},
            },
        )

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "analyze_vocabulary":
                function_args = json.loads(tool_call.function.arguments)

                structured_data = function_args
                formatted_result = format_vocabulary_result(function_args)

                return {"formatted": formatted_result, "structured": structured_data}

        return {
            "formatted": f"Không thể phân tích từ '{word}'. Vui lòng thử lại.",
            "structured": None,
        }

    except Exception as e:
        print(f"Error in explain_word: {e}")
        return {
            "formatted": f"Đã xảy ra lỗi khi phân tích từ '{word}': {str(e)}",
            "structured": None,
        }


def analyze_text_for_vocabulary(text):
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an English vocabulary tutor. "
                    "Extract 5-10 important vocabulary words from the given text passage. "
                    "Focus on words that are useful for English learners - intermediate to advanced level words, "
                    "excluding very basic words like 'the', 'and', 'is', etc. "
                    "For each word, provide Vietnamese meaning, examples, and learning tips."
                ),
            },
            {
                "role": "user",
                "content": f"Please extract important vocabulary words from this text and analyze each one:\n\n{text}",
            },
        ]

        response = client.chat.completions.create(
            model="GPT-4o-mini",
            messages=messages,
            tools=[TEXT_ANALYSIS_FUNCTION],
            tool_choice={
                "type": "function",
                "function": {"name": "extract_vocabulary_from_text"},
            },
        )

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "extract_vocabulary_from_text":
                function_args = json.loads(tool_call.function.arguments)
                vocabulary_list = function_args.get("vocabulary_list", [])

                results = []
                for vocab_data in vocabulary_list:
                    formatted_result = format_vocabulary_result(vocab_data)
                    results.append(
                        {
                            "word": vocab_data.get("word", ""),
                            "formatted": formatted_result,
                            "structured": vocab_data,
                        }
                    )

                return results

        return []

    except Exception as e:
        print(f"Error in analyze_text_for_vocabulary: {e}")
        return []


def extract_word_from_input(user_input):
    """
    Extract the actual English word from user input phrases like:
    - "help me add world hello" -> "hello"
    - "giúp tôi thêm từ hello" -> "hello"
    - "add word computer" -> "computer"
    - "thêm từ computer" -> "computer"
    - "hello" -> "hello" (direct word)
    """
    # Remove common phrases and extract the word
    user_input = user_input.lower().strip()

    # Common English phrases to remove
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
        "add word",
        "add world",
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

    # Common Vietnamese phrases to remove
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

    # Remove all common phrases
    for phrase in english_phrases + vietnamese_phrases:
        user_input = user_input.replace(phrase, "").strip()

    # Clean up extra spaces and punctuation
    user_input = re.sub(r"\s+", " ", user_input).strip()
    user_input = re.sub(r"[^\w\s]", "", user_input).strip()

    # If we still have multiple words, try to find the most likely English word
    words = user_input.split()

    if len(words) == 1:
        return words[0]
    elif len(words) > 1:
        # Look for the most likely English word (usually the last meaningful word)
        # Filter out common filler words
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

        # Remove filler words and get the last meaningful word
        meaningful_words = [word for word in words if word.lower() not in filler_words]

        if meaningful_words:
            # Prefer longer words as they're more likely to be vocabulary words
            meaningful_words.sort(key=len, reverse=True)
            return meaningful_words[0]  # Return the longest meaningful word
        else:
            return words[-1]  # If no meaningful words found, return the last word

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
    history = history[::-1]  # Đảo ngược để mới nhất ở đầu

    return render_template("index.html", result=result, word=word, history=history)


@app.route("/chat")
def chat():
    """Trang chatbox"""
    history = get_history()
    return render_template("chat.html", total_words=len(history))


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """API xử lý tin nhắn chat"""
    data = request.get_json()
    message = data.get("message", "").strip()
    chat_type = data.get("type", "")

    if not message:
        return jsonify({"success": False, "message": "Vui lòng nhập nội dung"})

    try:
        if chat_type == "word":
            # Extract the word from the user's input
            word_to_analyze = extract_word_from_input(message)
            print(f"User input: '{message}' -> Extracted word: '{word_to_analyze}'")

            result_data = explain_word(word_to_analyze)
            if result_data["structured"]:
                # Kiểm tra duplicate trước khi lưu
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
    """API xử lý tìm kiếm từ vựng bằng semantic search với function calling"""
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"success": False, "message": "Vui lòng nhập nội dung tìm kiếm"})

    try:
        # Sử dụng AI để quyết định có cần tìm kiếm từ vựng không và trích xuất query
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là một trợ lý học từ vựng tiếng Anh thông minh sử dụng semantic search. "
                    "Khi người dùng hỏi về từ vựng liên quan đến một chủ đề, khái niệm, tình huống, hoặc cảm xúc "
                    "(ví dụ: 'các từ liên quan đến du lịch', 'từ vựng về công nghệ', 'từ về cảm xúc vui buồn', 'words about happiness'), "
                    "hãy sử dụng function semantic_search_vocabulary để tìm kiếm trong cơ sở dữ liệu. "
                    "Semantic search sẽ tìm từ vựng dựa trên ý nghĩa và ngữ cảnh, không chỉ khớp từ khóa. "
                    "Nếu không phải câu hỏi về tìm kiếm từ vựng, hãy trả lời bình thường."
                ),
            },
            {"role": "user", "content": message},
        ]

        response = client.chat.completions.create(
            model="GPT-4o-mini",
            messages=messages,
            tools=[SEMANTIC_VOCABULARY_SEARCH_FUNCTION],
            tool_choice="auto",
        )

        # Kiểm tra xem AI có gọi function không
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "semantic_search_vocabulary":
                function_args = json.loads(tool_call.function.arguments)
                query = function_args.get("query", "")
                limit = function_args.get("limit", 10)
                similarity_threshold = function_args.get("similarity_threshold", 0.3)

                # Gọi function semantic search
                search_result = semantic_search_vocabulary_function(
                    query, limit, similarity_threshold
                )

                if search_result["success"]:
                    # Format kết quả để hiển thị với similarity scores
                    formatted_response = f"🔍 **Tìm thấy {search_result['count']} từ vựng liên quan đến '{query}' (Semantic Search):**\n\n"

                    for i, word_data in enumerate(search_result["results"], 1):
                        similarity_emoji = (
                            "🎯"
                            if word_data["similarity_score"] >= 0.7
                            else "📍" if word_data["similarity_score"] >= 0.5 else "📌"
                        )
                        formatted_response += f"**{i}. {word_data['word'].upper()}** /{word_data['phonetic']}/ ({word_data['part_of_speech']}) {similarity_emoji} {word_data['similarity_score']}\n"
                        formatted_response += (
                            f"   📝 {word_data['vietnamese_meaning']}\n"
                        )
                        formatted_response += (
                            f"   🏷️ Category: {word_data['category']}\n"
                        )
                        if word_data["example_sentences"]:
                            formatted_response += (
                                f"   💡 Ví dụ: {word_data['example_sentences'][0]}\n"
                            )
                        formatted_response += "\n"

                    return jsonify(
                        {
                            "success": True,
                            "message": formatted_response,
                            "type": "semantic_search",
                            "query": query,
                            "results": search_result["results"],
                            "count": search_result["count"],
                        }
                    )
                else:
                    return jsonify(
                        {
                            "success": False,
                            "message": search_result["message"],
                            "type": "semantic_search",
                        }
                    )
        else:
            # Không có function call, trả lời bình thường
            ai_response = response.choices[0].message.content
            return jsonify(
                {"success": True, "message": ai_response, "type": "general_chat"}
            )

    except Exception as e:
        return jsonify({"success": False, "message": f"❌ Đã xảy ra lỗi: {str(e)}"})


@app.route("/list")
def flashcard_list():
    """Hiển thị danh sách tất cả flashcard"""
    history = get_history()
    history = history[::-1]  # Đảo ngược để mới nhất ở đầu
    return render_template("list.html", history=history)


@app.route("/delete/<int:index>")
def delete_flashcard_route(index):
    """Xóa flashcard theo index"""
    history = get_history()
    actual_index = len(history) - 1 - index
    delete_flashcard(actual_index)
    return redirect(url_for("flashcard_list"))


@app.route("/api/delete-word", methods=["POST"])
def delete_word_api():
    """API xóa từ vựng theo tên từ"""
    try:
        data = request.get_json()
        word = data.get("word", "").strip()

        if not word:
            return jsonify(
                {"success": False, "message": "Vui lòng cung cấp tên từ cần xóa"}
            )

        # Xóa từ ChromaDB
        deleted_count = vocab_manager.delete_vocabulary(word)

        # Xóa từ history.json
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
    """API xóa toàn bộ dữ liệu từ ChromaDB và history.json"""
    try:
        # Xác nhận từ client
        data = request.get_json()
        confirm = data.get("confirm", False)

        if not confirm:
            return jsonify(
                {
                    "success": False,
                    "message": "Vui lòng xác nhận bằng cách gửi 'confirm': true",
                }
            )

        # Thực hiện xóa
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
    """API endpoint để lấy thông tin từ dưới dạng JSON"""
    result_data = explain_word(word)
    return {
        "word": word,
        "formatted_result": result_data["formatted"],
        "structured_data": result_data["structured"],
        "audio_path": result_data.get("audio_path"),
    }


# @app.route("/api/search-topic", methods=["POST"])
# def search_by_topic_api():
#     """API tìm kiếm từ vựng theo chủ đề - ĐÃ XÓA"""
#     return jsonify({"success": False, "message": "Tính năng tìm kiếm thông minh đã bị xóa"})


@app.route("/api/search-category", methods=["POST"])
def search_by_category_api():
    """API tìm kiếm từ vựng theo category"""
    try:
        data = request.get_json()
        category = data.get("category", "").strip()
        limit = data.get("limit", 50)

        if not category:
            return jsonify({"success": False, "message": "Vui lòng chọn category"})

        # Tìm kiếm trong ChromaDB theo category
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
    """API lấy thống kê các category"""
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
    """Trang tìm kiếm thông minh với semantic search"""
    return render_template("search.html")


@app.route("/api/generate-audio/<word>")
def generate_audio_api(word):
    """Generate or get existing audio for a word"""
    try:
        # Find word in history
        history = get_history()
        for item in history:
            if item["word"].lower() == word.lower():
                result = item.get("result", {})

                # If audio already exists, return it
                if result.get("audio_path"):
                    return jsonify(
                        {
                            "success": True,
                            "audio_path": result["audio_path"],
                            "word": word,
                        }
                    )

                # Generate new audio if structured data exists
                if result.get("structured") and result["structured"].get(
                    "vietnamese_meaning"
                ):
                    vietnamese_meaning = result["structured"]["vietnamese_meaning"]
                    audio_path = tts_service.generate_audio(word, vietnamese_meaning)
                    if audio_path:
                        # Update history with audio path
                        result["audio_path"] = audio_path
                        result["audio_text"] = f"{word}. {vietnamese_meaning}."

                        # Save updated history
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


# Serve static files
@app.route("/static/audio/<filename>")
def serve_audio(filename):
    """Serve audio files"""
    return send_from_directory("static/audio", filename)


if __name__ == "__main__":
    app.run(debug=True, port=6969)
