#!/usr/bin/env python3
"""
Demo script to test the intelligent vocabulary agent
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import vocab_agent


def demo_intelligent_chat():
    """Demo the intelligent chat functionality"""

    print("🤖 Intelligent Vocabulary Agent Demo")
    print("=" * 50)
    print()

    # Test cases for different intents
    test_cases = [
        "Xin chào! Bạn có thể giúp tôi học từ vựng không?",
        "Tìm từ vựng về du lịch",
        "Thêm từ beautiful vào flashcard",
        "Có bao nhiêu từ trong database?",
        "Phân tích đoạn văn này: I love traveling to different countries and experiencing new cultures",
        "words about technology",
    ]

    for i, test_message in enumerate(test_cases, 1):
        print(f"🔍 Test Case {i}: {test_message}")
        print("-" * 30)

        try:
            response = vocab_agent.invoke({"input": test_message})

            if isinstance(response, dict):
                output = response.get("output", str(response))
            else:
                output = str(response)

            print(f"🤖 Response: {output}")
            print()

        except Exception as e:
            print(f"❌ Error: {e}")
            print()

    print("✅ Demo completed!")


if __name__ == "__main__":
    demo_intelligent_chat()
