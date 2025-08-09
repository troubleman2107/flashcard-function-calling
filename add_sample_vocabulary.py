#!/usr/bin/env python3
"""
Script để thêm từ vựng mẫu cho các categories
"""

import requests
import json
import time

# API endpoint
API_URL = "http://localhost:6969/api/chat"

# Sample vocabulary data for different categories
SAMPLE_VOCABULARY = {
    "Travel": [
        "vacation", "hotel", "passport", "luggage", "destination"
    ],
    "Technology": [
        "software", "internet", "smartphone", "database", "algorithm"
    ],
    "Food": [
        "restaurant", "delicious", "recipe", "ingredient", "nutrition"
    ],
    "Business": [
        "meeting", "profit", "customer", "marketing", "investment"
    ],
    "Health": [
        "exercise", "medicine", "hospital", "doctor", "healthy"
    ],
    "Education": [
        "student", "teacher", "university", "knowledge", "learning"
    ],
    "Sports": [
        "football", "training", "competition", "athlete", "victory"
    ],
    "Entertainment": [
        "movie", "music", "concert", "theater", "performance"
    ],
    "Nature": [
        "forest", "mountain", "ocean", "wildlife", "environment"
    ],
    "Family": [
        "parents", "children", "relatives", "wedding", "birthday"
    ]
}

def add_word_to_flashcard(word):
    """Thêm một từ vào flashcard thông qua API"""
    try:
        payload = {
            "message": word,
            "type": "word"
        }
        
        response = requests.post(API_URL, json=payload, timeout=30)
        data = response.json()
        
        if data.get("success"):
            print(f"✅ Đã thêm từ '{word}' thành công!")
            return True
        else:
            print(f"❌ Lỗi thêm từ '{word}': {data.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Exception khi thêm từ '{word}': {str(e)}")
        return False

def add_category_words(category, words):
    """Thêm tất cả từ vựng của một category"""
    print(f"\n🏷️ Đang thêm từ vựng cho category: {category}")
    print("=" * 50)
    
    success_count = 0
    for word in words:
        if add_word_to_flashcard(word):
            success_count += 1
        
        # Delay để tránh spam API
        time.sleep(2)
    
    print(f"\n📊 Kết quả category {category}: {success_count}/{len(words)} từ thành công")
    return success_count

def main():
    """Main function"""
    print("🚀 Bắt đầu thêm từ vựng mẫu cho các categories...")
    print("⏰ Quá trình này có thể mất vài phút...")
    
    total_success = 0
    total_words = 0
    
    for category, words in SAMPLE_VOCABULARY.items():
        success_count = add_category_words(category, words)
        total_success += success_count
        total_words += len(words)
        
        # Delay giữa các categories
        time.sleep(3)
    
    print("\n" + "=" * 60)
    print(f"🎉 HOÀN THÀNH!")
    print(f"📈 Tổng kết: {total_success}/{total_words} từ được thêm thành công")
    print(f"📊 Tỷ lệ thành công: {(total_success/total_words)*100:.1f}%")
    
    if total_success > 0:
        print(f"\n✨ Bây giờ bạn có thể test semantic search với:")
        print("   - 'từ về du lịch' (Travel)")
        print("   - 'công nghệ máy tính' (Technology)")
        print("   - 'từ vựng ăn uống' (Food)")
        print("   - 'business terms' (Business)")
        print("   - 'health and medical' (Health)")

if __name__ == "__main__":
    main()