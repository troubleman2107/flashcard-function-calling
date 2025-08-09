from flask import Flask, render_template, request, redirect, url_for, jsonify
import openai
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Khởi tạo OpenAI client
client = openai.OpenAI(
    base_url=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
)

HISTORY_FILE = "history.json"

# Định nghĩa function schema cho OpenAI
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


# Hàm lưu kết quả vào file JSON
def save_to_history(word, result):
    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                history_data = json.load(f)
            except json.JSONDecodeError:
                pass

    history_data.append({"word": word, "result": result})

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)


# Hàm lấy danh sách flashcard
def get_history():
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass
    return history


# Hàm xóa flashcard
def delete_flashcard(index):
    history_data = get_history()
    if 0 <= index < len(history_data):
        del history_data[index]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)


# Hàm format kết quả từ function calling
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

    # Tạo formatted string
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


# Hàm gọi API với function calling cho từ đơn
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

        # Xử lý response từ function calling
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "analyze_vocabulary":
                function_args = json.loads(tool_call.function.arguments)

                # Lưu cả structured data và formatted text
                structured_data = function_args
                formatted_result = format_vocabulary_result(function_args)

                # Trả về cả hai dạng dữ liệu
                return {"formatted": formatted_result, "structured": structured_data}

        # Fallback nếu function calling không hoạt động
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


# Hàm phân tích văn bản và trích xuất từ vựng
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

        # Xử lý response từ function calling
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            if tool_call.function.name == "extract_vocabulary_from_text":
                function_args = json.loads(tool_call.function.arguments)
                vocabulary_list = function_args.get("vocabulary_list", [])

                # Format mỗi từ và trả về danh sách
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
            # Xử lý thêm từ đơn
            result_data = explain_word(message)
            if result_data["structured"]:
                save_to_history(message, result_data)
                return jsonify(
                    {
                        "success": True,
                        "message": f"✅ Đã thêm từ '{message}' vào flashcard!",
                        "result": result_data["formatted"],
                        "structured_data": result_data["structured"],
                        "type": "word",
                    }
                )
            else:
                return jsonify(
                    {
                        "success": False,
                        "message": f"❌ Không thể phân tích từ '{message}'",
                    }
                )

        elif chat_type == "text":
            # Xử lý phân tích văn bản
            vocabulary_results = analyze_text_for_vocabulary(message)
            if vocabulary_results:
                # Lưu tất cả từ vào history
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
    # Chuyển đổi index vì list được đảo ngược khi hiển thị
    actual_index = len(history) - 1 - index
    delete_flashcard(actual_index)
    return redirect(url_for("flashcard_list"))


@app.route("/api/word/<word>")
def get_word_api(word):
    """API endpoint để lấy thông tin từ dưới dạng JSON"""
    result_data = explain_word(word)
    return {
        "word": word,
        "formatted_result": result_data["formatted"],
        "structured_data": result_data["structured"],
    }


if __name__ == "__main__":
    app.run(debug=True)
