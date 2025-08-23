#!/usr/bin/env python3
"""
Comprehensive Test Runner for Flashcard System
Runs both original test cases and new workflow flow tests, generating detailed HTML report
"""

import unittest
import sys
import os
import time
from datetime import datetime
import subprocess
import ast
import inspect

def extract_test_source_code():
    """Extract actual source code and details from test files"""
    test_source_info = {}
    
    # Test files to analyze
    test_files = ['test_cases.py', 'test_workflow_flows.py']
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            continue
            
        try:
            # Read the test file
            with open(test_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse the AST
            tree = ast.parse(source_code)
            
            # Find test classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.endswith('Test'):
                    # Extract each test method
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and method.name.startswith('test_'):
                            # Get the docstring
                            docstring = ast.get_docstring(method)
                            
                            # Get the source code lines for this method
                            method_source_lines = source_code.split('\n')[method.lineno-1:method.end_lineno]
                            method_source = '\n'.join(method_source_lines)
                            
                            # Extract key information from the method
                            test_source_info[method.name] = {
                                'docstring': docstring or 'No documentation',
                                'source_code': method_source,
                                'line_start': method.lineno,
                                'line_end': method.end_lineno,
                                'file': test_file
                            }
        except Exception as e:
            print(f"Error extracting test source from {test_file}: {e}")
    
    return test_source_info

def generate_html_report(test_result, execution_time):
    """Generate detailed HTML report with test descriptions and comprehensive results"""
    
    # Extract actual test source code
    test_source_info = extract_test_source_code()
    
    # Test descriptions mapping with detailed information
    test_descriptions = {
        # Original test cases
        'test_vocabulary_word_model_validation': {
            'title': 'Kiểm tra validation của Pydantic model VocabularyWord',
            'description': 'Test case này kiểm tra việc tạo và validate VocabularyWord model với tất cả các field bắt buộc',
            'input': 'VocabularyWord(word="hello", vietnamese_meaning="xin chào", part_of_speech="interjection", phonetic="/həˈloʊ/", example_sentences=["Hello, how are you?"], mnemonic_tip="Hello = xin chào", difficulty_level="beginner", synonyms=["hi", "greetings"])',
            'expected': 'Model được tạo thành công với word="hello", vietnamese_meaning="xin chào", part_of_speech="interjection"',
            'category': 'Data Models'
        },
        'test_vocabulary_list_model': {
            'title': 'Kiểm tra VocabularyList model cho batch processing',
            'description': 'Test case này kiểm tra VocabularyList model có thể chứa nhiều VocabularyWord objects',
            'input': 'VocabularyList với 2 words: [word1="first", word2="second"]',
            'expected': 'VocabularyList chứa 2 words, có thể iterate qua từng word',
            'category': 'Data Models'
        },
        'test_tts_service_initialization': {
            'title': 'Kiểm tra khởi tạo TTS service',
            'description': 'Test case này kiểm tra việc khởi tạo Text-to-Speech service và loading models',
            'input': 'TTSService() constructor call',
            'expected': 'Service khởi tạo thành công, models được load từ Hugging Face',
            'category': 'Audio Services'
        },
        'test_tts_audio_generation': {
            'title': 'Kiểm tra việc tạo audio từ text',
            'description': 'Test case này kiểm tra quá trình tạo audio file từ text input sử dụng TTS models',
            'input': 'Text: "hello", Vietnamese meaning: "xin chào"',
            'expected': 'Audio file được tạo thành công, path trả về format "/static/audio/..."',
            'category': 'Audio Services'
        },
        'test_vocabulary_manager_initialization': {
            'title': 'Kiểm tra khởi tạo VocabularyManager',
            'description': 'Test case này kiểm tra việc khởi tạo VocabularyManager với ChromaDB backend',
            'input': 'VocabularyManager(persist_directory="./chroma_db")',
            'expected': 'Manager khởi tạo thành công, ChromaDB connection established',
            'category': 'Data Management'
        },
        'test_vocabulary_manager_add_vocabulary': {
            'title': 'Kiểm tra thêm từ vựng mới',
            'description': 'Test case này kiểm tra việc thêm từ vựng mới vào VocabularyManager',
            'input': 'VocabularyWord(word="innovation", vietnamese_meaning="sáng kiến", part_of_speech="noun", phonetic="/ˌɪnəˈveɪʃən/", example_sentences=["Innovation is key"], mnemonic_tip="Innovation = sáng kiến", difficulty_level="advanced", synonyms=["creativity", "invention"])',
            'expected': 'Từ vựng được thêm thành công, category được classify tự động, lưu vào ChromaDB',
            'category': 'Data Management'
        },
        'test_vocabulary_manager_search_by_category': {
            'title': 'Kiểm tra tìm kiếm theo category',
            'description': 'Test case này kiểm tra khả năng tìm kiếm từ vựng theo category cụ thể',
            'input': 'Category: "Technology", limit: 5',
            'expected': 'Danh sách từ vựng thuộc category Technology, đúng format và số lượng',
            'category': 'Data Management'
        },
        'test_semantic_search_with_chromadb': {
            'title': 'Kiểm tra semantic search với ChromaDB',
            'description': 'Test case này kiểm tra semantic search sử dụng ChromaDB embeddings',
            'input': 'Query: "du lịch" (tiếng Việt)',
            'expected': 'Kết quả semantic search với similarity scores, xử lý lỗi API gracefully',
            'category': 'Search & Retrieval'
        },
        'test_fallback_keyword_search': {
            'title': 'Kiểm tra fallback keyword search',
            'description': 'Test case này kiểm tra fallback mechanism khi semantic search thất bại',
            'input': 'Mock ChromaDB collection với test data: word="business", category="Business"',
            'expected': 'Kết quả keyword search từ collection data: 1 result với word="business"',
            'category': 'Search & Retrieval'
        },
        'test_fallback_keyword_search_on_semantic_failure': {
            'title': 'Kiểm tra xử lý graceful khi semantic search thất bại',
            'description': 'Test case này kiểm tra việc xử lý graceful khi semantic search gặp lỗi',
            'input': 'Mocked semantic search failure với query "test query"',
            'expected': 'Hệ thống xử lý lỗi gracefully, trả về empty results []',
            'category': 'Error Handling'
        },
        'test_langchain_setup': {
            'title': 'Kiểm tra khởi tạo LangChain framework',
            'description': 'Test case này kiểm tra việc setup LangChain framework với các tools cần thiết',
            'input': 'setup_langchain() function call',
            'expected': 'Framework được khởi tạo thành công với tools và memory system',
            'category': 'AI Framework'
        },
        'test_vocabulary_agent_creation': {
            'title': 'Kiểm tra tạo vocabulary agent',
            'description': 'Test case này kiểm tra việc tạo vocabulary agent với LangChain',
            'input': 'create_vocabulary_agent() function call',
            'expected': 'Agent được tạo thành công với đầy đủ tools, memory, và capabilities',
            'category': 'AI Framework'
        },
        'test_explain_word_with_langchain': {
            'title': 'Kiểm tra giải thích từ vựng với LangChain',
            'description': 'Test case này kiểm tra việc sử dụng LangChain để giải thích từ vựng',
            'input': 'Word: "hello", mock LangChain response',
            'expected': 'Structured vocabulary explanation với đầy đủ fields: word, meaning, examples, tips',
            'category': 'AI Framework'
        },
        'test_analyze_text_for_vocabulary': {
            'title': 'Kiểm tra phân tích text để trích xuất từ vựng',
            'description': 'Test case này kiểm tra việc phân tích text input để trích xuất từ vựng mới',
            'input': 'Text: "This is a sample text with multiple words for vocabulary extraction"',
            'expected': 'List các VocabularyWord objects được trích xuất từ text',
            'category': 'AI Framework'
        },
        'test_format_vocabulary_result': {
            'title': 'Kiểm tra format kết quả từ vựng',
            'description': 'Test case này kiểm tra việc format kết quả từ vựng từ LangChain output',
            'input': 'Raw LangChain function output với vocabulary data',
            'expected': 'Formatted vocabulary data với cấu trúc chuẩn cho UI display',
            'category': 'Data Processing'
        },
        'test_save_to_history': {
            'title': 'Kiểm tra lưu từ vựng vào history',
            'description': 'Test case này kiểm tra việc lưu từ vựng vào history system và ChromaDB',
            'input': 'Word: "test", vocabulary result object',
            'expected': 'Từ vựng được lưu vào history file và ChromaDB, category được classify',
            'category': 'Data Management'
        },
        'test_extract_word_from_input': {
            'title': 'Kiểm tra trích xuất từ vựng từ user input',
            'description': 'Test case này kiểm tra logic trích xuất từ vựng từ user input text',
            'input': 'User inputs: ["help me add word hello", "giúp tôi thêm từ xin chào", "thêm từ vựng mới"]',
            'expected': 'Extracted words: ["hello", "chào", "vựng"]',
            'category': 'Text Processing'
        },
        'test_get_history': {
            'title': 'Kiểm tra lấy lịch sử từ vựng',
            'description': 'Test case này kiểm tra việc đọc và trả về lịch sử từ vựng đã học',
            'input': 'History file path với sample data',
            'expected': 'List các từ vựng đã học với đầy đủ thông tin, bao gồm word "test"',
            'category': 'Data Management'
        },
        'test_word_exists_in_history': {
            'title': 'Kiểm tra kiểm tra từ vựng đã tồn tại',
            'description': 'Test case này kiểm tra việc kiểm tra xem một từ vựng đã tồn tại trong history hay chưa',
            'input': 'Word: "test" để kiểm tra trong history',
            'expected': 'True vì từ "test" đã tồn tại trong history',
            'category': 'Data Management'
        },
        'test_get_categories_stats': {
            'title': 'Kiểm tra thống kê theo category',
            'description': 'Test case này kiểm tra việc tính toán thống kê số lượng từ vựng theo từng category',
            'input': 'Vocabulary data từ ChromaDB với multiple categories',
            'expected': 'Dictionary với category names và counts, ví dụ: {"Technology": 2, "Education": 1}',
            'category': 'Data Analytics'
        },
        'test_error_handling_in_word_analysis': {
            'title': 'Kiểm tra xử lý lỗi trong word analysis',
            'description': 'Test case này kiểm tra việc xử lý lỗi khi word analysis gặp vấn đề',
            'input': 'Mocked error condition: LangChain error',
            'expected': 'Error được log và xử lý gracefully, không crash, trả về error message',
            'category': 'Error Handling'
        },
        'test_end_to_end_vocabulary_workflow': {
            'title': 'Kiểm tra workflow hoàn chỉnh từ input đến storage',
            'description': 'Test case này kiểm tra toàn bộ workflow từ việc nhận input từ user đến khi lưu vào storage',
            'input': 'Complete user input workflow với word "integration"',
            'expected': 'Từ vựng được xử lý hoàn chỉnh: analyze → classify → save, category: Technology',
            'category': 'Integration'
        },
        'test_multi_language_support': {
            'title': 'Kiểm tra hỗ trợ đa ngôn ngữ',
            'description': 'Test case này kiểm tra khả năng xử lý từ vựng từ nhiều ngôn ngữ khác nhau',
            'input': 'Multi-language inputs: English "hello", Vietnamese "xin chào"',
            'expected': 'Xử lý thành công cả tiếng Anh và tiếng Việt, tạo VocabularyWord objects',
            'category': 'Internationalization'
        },
        'test_concurrent_vocabulary_processing': {
            'title': 'Kiểm tra xử lý đồng thời nhiều từ vựng',
            'description': 'Test case này kiểm tra khả năng xử lý nhiều từ vựng cùng lúc một cách hiệu quả',
            'input': '5 vocabulary words cùng lúc: word0, word1, word2, word3, word4',
            'expected': 'Tất cả 5 từ vựng được xử lý thành công, không có conflict',
            'category': 'Performance'
        },
        'test_memory_efficiency': {
            'title': 'Kiểm tra hiệu quả sử dụng memory',
            'description': 'Test case này kiểm tra hiệu quả sử dụng memory trong conversation handling',
            'input': 'Multiple conversation turns với vocabulary queries',
            'expected': 'Memory usage ổn định, conversation context được maintain đúng cách',
            'category': 'Performance'
        },
        
        # New workflow flow tests
        'test_faq_question_flow': {
            'title': 'Test FAQ Question Flow - Complete workflow from FAQ input to answer',
            'description': 'Test case này kiểm tra luồng hoạt động hoàn chỉnh từ FAQ input đến câu trả lời',
            'input': 'FAQ Question: "Làm thế nào để thêm từ vựng mới?"',
            'expected': 'FAQ answer được trả về với format chuẩn, source và confidence score',
            'category': 'Workflow Flows'
        },
        'test_faq_question_flow_with_unknown_question': {
            'title': 'Test FAQ Question Flow with unknown question - should return default response',
            'description': 'Test case này kiểm tra luồng FAQ với câu hỏi không có trong database',
            'input': 'FAQ Question: "Câu hỏi không có trong database?"',
            'expected': 'Default response được trả về với thông báo xin lỗi',
            'category': 'Workflow Flows'
        },
        'test_chat_message_flow': {
            'title': 'Test Chat Message Flow - Complete workflow from chat input to response',
            'description': 'Test case này kiểm tra luồng hoạt động hoàn chỉnh từ chat input đến response',
            'input': 'Chat Message: "Xin chào, bạn có thể giúp tôi học từ vựng không?"',
            'expected': 'Chat response với suggestions và context được trả về',
            'category': 'Workflow Flows'
        },
        'test_chat_message_flow_with_vocabulary_request': {
            'title': 'Test Chat Message Flow with vocabulary learning request',
            'description': 'Test case này kiểm tra luồng chat với yêu cầu học từ vựng',
            'input': 'Chat Message: "Tôi muốn thêm từ innovation vào danh sách từ vựng"',
            'expected': 'Chat response với action="add_vocabulary" và word="innovation"',
            'category': 'Workflow Flows'
        },
        'test_text_passage_flow': {
            'title': 'Test Text Passage Flow - Complete workflow from text input to vocabulary extraction',
            'description': 'Test case này kiểm tra luồng hoạt động hoàn chỉnh từ text input đến vocabulary extraction',
            'input': 'Text Passage: "Artificial intelligence is transforming the world. Human intelligence combined with AI creates powerful solutions."',
            'expected': 'List các VocabularyWord objects được trích xuất từ text: artificial, intelligence',
            'category': 'Workflow Flows'
        },
        'test_text_passage_flow_with_empty_text': {
            'title': 'Test Text Passage Flow with empty text - should return error',
            'description': 'Test case này kiểm tra luồng text passage với text rỗng',
            'input': 'Text Passage: "" (empty string)',
            'expected': 'Error 400 với message "Text cannot be empty"',
            'category': 'Workflow Flows'
        },
        'test_search_query_flow': {
            'title': 'Test Search Query Flow - Complete workflow from search query to results',
            'description': 'Test case này kiểm tra luồng hoạt động hoàn chỉnh từ search query đến results',
            'input': 'Search Query: "workflow" với limit=5',
            'expected': 'Search results với word="workflow", similarity_score=0.89, category="Business"',
            'category': 'Workflow Flows'
        },
        'test_search_query_flow_with_category_filter': {
            'title': 'Test Search Query Flow with category filter',
            'description': 'Test case này kiểm tra luồng search với category filter',
            'input': 'Search Query: Category "Technology" với limit=5',
            'expected': 'Search results chỉ với category="Technology", ví dụ: word="innovation"',
            'category': 'Workflow Flows'
        },
        'test_single_word_flow': {
            'title': 'Test Single Word Flow - Complete workflow from single word to detailed analysis',
            'description': 'Test case này kiểm tra luồng hoạt động hoàn chỉnh từ single word đến detailed analysis',
            'input': 'Single Word: "efficiency"',
            'expected': 'Word explanation với đầy đủ fields: meaning="hiệu quả", examples, tips, synonyms',
            'category': 'Workflow Flows'
        },
        'test_single_word_flow_with_audio_generation': {
            'title': 'Test Single Word Flow with audio generation',
            'description': 'Test case này kiểm tra luồng single word với audio generation',
            'input': 'Single Word: "efficiency" với generate_audio=True',
            'expected': 'Word explanation + audio_path="/static/audio/efficiency_12345.wav"',
            'category': 'Workflow Flows'
        },
        'test_integrated_workflow_from_chat_to_vocabulary': {
            'title': 'Test integrated workflow: Chat → Word Analysis → Save to History',
            'description': 'Test case này kiểm tra workflow tích hợp: Chat → Word Analysis → Save to History',
            'input': 'Integrated workflow: Chat message → Word "collaboration" → Save to history',
            'expected': '3 steps completed: chat response, word explanation, save to history',
            'category': 'Integration Flows'
        },
        'test_workflow_error_handling': {
            'title': 'Test workflow error handling across different flows',
            'description': 'Test case này kiểm tra error handling across different workflow flows',
            'input': 'Error conditions: FAQ service error, Chat service error, Search error',
            'expected': 'All errors handled gracefully với HTTP 500 status và error messages',
            'category': 'Error Handling'
        },
        'test_concurrent_workflow_processing': {
            'title': 'Test concurrent processing of multiple workflow requests',
            'description': 'Test case này kiểm tra concurrent processing của multiple workflow requests',
            'input': '9 concurrent requests: 3 chat + 3 search + 3 word analysis',
            'expected': 'All 9 requests processed successfully trong <10 seconds, no errors',
            'category': 'Performance & Scalability'
        }
    }
    
    # Process test results
    test_details = []
    
    # Get all test names from the test suite
    all_test_names = set(test_descriptions.keys())
    
    # Process successful tests (tests that ran without failures or errors)
    successful_tests = all_test_names - set([test for test, _ in test_result.failures]) - set([test for test, _ in test_result.errors])
    
    for test_name in successful_tests:
        test_info = test_descriptions.get(test_name, {})
        source_info = test_source_info.get(test_name, {})
        
        test_details.append({
            'name': test_name,
            'title': test_info.get('title', f'Test: {test_name}'),
            'description': test_info.get('description', source_info.get('docstring', 'Không có mô tả')),
            'input': test_info.get('input', 'Xem source code để biết chi tiết'),
            'expected': test_info.get('expected', 'Xem source code để biết chi tiết'),
            'category': test_info.get('category', 'General'),
            'source_code': source_info.get('source_code', 'Không tìm thấy source code'),
            'line_range': f"Lines {source_info.get('line_start', 'N/A')}-{source_info.get('line_end', 'N/A')} in {source_info.get('file', 'N/A')}",
            'status': 'PASSED',
            'result': '✅ Thành công',
            'details': 'PASSED'
        })
    
    # Process failures
    for test, traceback in test_result.failures:
        test_info = test_descriptions.get(test, {})
        source_info = test_source_info.get(test, {})
        
        test_details.append({
            'name': test,
            'title': test_info.get('title', f'Test: {test}'),
            'description': test_info.get('description', source_info.get('docstring', 'Không có mô tả')),
            'input': test_info.get('input', 'Xem source code để biết chi tiết'),
            'expected': test_info.get('expected', 'Xem source code để biết chi tiết'),
            'category': test_info.get('category', 'General'),
            'source_code': source_info.get('source_code', 'Không tìm thấy source code'),
            'line_range': f"Lines {source_info.get('line_start', 'N/A')}-{source_info.get('line_end', 'N/A')} in {source_info.get('file', 'N/A')}",
            'status': 'FAILED',
            'result': '❌ Thất bại',
            'details': f'<pre>{traceback}</pre>'
        })
    
    # Process errors
    for test, traceback in test_result.errors:
        test_info = test_descriptions.get(test, {})
        source_info = test_source_info.get(test, {})
        
        test_details.append({
            'name': test,
            'title': test_info.get('title', f'Test: {test}'),
            'description': test_info.get('description', source_info.get('docstring', 'Không có mô tả')),
            'input': test_info.get('input', 'Xem source code để biết chi tiết'),
            'expected': test_info.get('expected', 'Xem source code để biết chi tiết'),
            'category': test_info.get('category', 'General'),
            'source_code': source_info.get('source_code', 'Không tìm thấy source code'),
            'line_range': f"Lines {source_info.get('line_start', 'N/A')}-{source_info.get('line_end', 'N/A')} in {source_info.get('file', 'N/A')}",
            'status': 'ERROR',
            'result': '💥 Lỗi',
            'details': f'<pre>{traceback}</pre>'
        })
    
    # Sort test details by status (errors first, then failures, then passed)
    status_order = {'ERROR': 0, 'FAILED': 1, 'PASSED': 2}
    test_details.sort(key=lambda x: status_order[x['status']])
    
    # Get test results
    tests_run = test_result.testsRun
    failures = len(test_result.failures)
    errors = len(test_result.errors)
    success_rate = ((tests_run - failures - errors) / tests_run * 100) if tests_run > 0 else 0
    
    html_content = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flashcard System Complete Test Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .execution-time {{
            background: #17a2b8;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 1.1em;
        }}
        
        .summary {{
            background: #f8f9fa;
            padding: 30px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-left: 4px solid #007bff;
        }}
        
        .summary-card.success {{
            border-left-color: #28a745;
        }}
        
        .summary-card.failure {{
            border-left-color: #dc3545;
        }}
        
        .summary-card.error {{
            border-left-color: #ffc107;
        }}
        
        .summary-card .number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .summary-card.success .number {{
            color: #28a745;
        }}
        
        .summary-card.failure .number {{
            color: #dc3545;
        }}
        
        .summary-card.error .number {{
            color: #ffc107;
        }}
        
        .summary-card .label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .test-results {{
            padding: 30px;
        }}
        
        .test-results h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }}
        
        .test-item {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .test-header {{
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #f8f9fa;
        }}
        
        .test-header.passed {{
            background: #d4edda;
            border-left: 4px solid #28a745;
        }}
        
        .test-header.failed {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
        }}
        
        .test-header.error {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        
        .test-name {{
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
        }}
        
        .test-category {{
            background: #6c757d;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .test-status {{
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .test-status.passed {{
            background: #28a745;
            color: white;
        }}
        
        .test-status.failed {{
            background: #dc3545;
            color: white;
        }}
        
        .test-status.error {{
            background: #ffc107;
            color: #212529;
        }}
        
        .test-title {{
            padding: 15px 20px;
            background: white;
            color: #2c3e50;
            font-weight: bold;
            font-size: 1.1em;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .test-description {{
            padding: 15px 20px;
            background: white;
            color: #6c757d;
            font-style: italic;
            border-bottom: 1px solid #e9ecef;
            line-height: 1.6;
        }}
        
        .test-input-expected {{
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .test-input-expected .section {{
            margin-bottom: 15px;
        }}
        
        .test-input-expected .section:last-child {{
            margin-bottom: 0;
        }}
        
        .test-input-expected .label {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 5px;
            display: block;
        }}
        
        .test-input-expected .content {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
            font-family: monospace;
            font-size: 0.9em;
        }}
        
        .test-details {{
            padding: 20px;
            background: white;
        }}
        
        .test-details pre {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 0.9em;
            border: 1px solid #e9ecef;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
            
            .test-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Flashcard System Complete Test Report</h1>
            <div class="subtitle">Báo cáo chi tiết kết quả test suite hoàn chỉnh bao gồm cả workflow flows</div>
        </div>
        
        <div class="execution-time">
            ⏱️ Thời gian thực thi: {execution_time:.2f} giây
        </div>
        
        <div class="summary">
            <h2>📊 Tổng quan kết quả</h2>
            <div class="summary-grid">
                <div class="summary-card success">
                    <div class="number">{tests_run - failures - errors}</div>
                    <div class="label">Tests Thành công</div>
                </div>
                <div class="summary-card failure">
                    <div class="number">{failures}</div>
                    <div class="label">Tests Thất bại</div>
                </div>
                <div class="summary-card error">
                    <div class="number">{errors}</div>
                    <div class="label">Tests Lỗi</div>
                </div>
                <div class="summary-card">
                    <div class="number">{success_rate:.1f}%</div>
                    <div class="label">Tỷ lệ thành công</div>
                </div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>🔍 Chi tiết từng test case</h2>
            
            {''.join([f'''
            <div class="test-item">
                <div class="test-header {test['status'].lower()}">
                    <div class="test-name">{test['name']}</div>
                    <div class="test-status {test['status'].lower()}">{test['result']}</div>
                </div>
                <div class="test-title">
                    <span class="test-category">{test['category']}</span> - {test['title']}
                </div>
                <div class="test-description">
                    📝 {test['description']}
                </div>
                <div class="test-input-expected">
                    <div class="section">
                        <label class="label">📥 Input Data:</label>
                        <div class="content">{test['input']}</div>
                    </div>
                    <div class="section">
                        <label class="label">🎯 Expected Output:</label>
                        <div class="content">{test['expected']}</div>
                    </div>
                </div>
                <div class="test-details">
                    <strong>📊 Kết quả thực tế:</strong> {test['details']}
                </div>
            </div>
            ''' for test in test_details])}
        </div>
        
        <div class="footer">
            <p>📅 Báo cáo được tạo vào: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            <p>🚀 Flashcard System - AI-Powered Vocabulary Learning với Workflow Flows</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content

def run_all_tests_with_report():
    """Run all tests and generate comprehensive HTML report"""
    print("🧪 Starting Complete Flashcard System Test Suite...")
    print("=" * 60)
    
    # Load test suites
    loader = unittest.TestLoader()
    
    # Load original test cases
    original_suite = loader.loadTestsFromName('test_cases.TestFlashcardSystem')
    
    # Load new workflow flow tests
    workflow_suite = loader.loadTestsFromName('test_workflow_flows.TestWorkflowFlows')
    
    # Combine test suites
    combined_suite = unittest.TestSuite([original_suite, workflow_suite])
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(combined_suite)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Generate HTML report
    html_report = generate_html_report(test_result, execution_time)
    
    # Save HTML report
    report_filename = f"complete_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    # Print summary
    print("=" * 60)
    print("📊 COMPLETE TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {test_result.testsRun}")
    print(f"Passed: {test_result.testsRun - len(test_result.failures) - len(test_result.errors)}")
    print(f"Failures: {len(test_result.failures)}")
    print(f"Errors: {len(test_result.errors)}")
    print(f"Success rate: {((test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100):.1f}%")
    print(f"Execution time: {execution_time:.2f} seconds")
    print()
    print(f"📄 HTML Report generated: {report_filename}")
    print("🌐 Open the HTML file in your browser to view detailed results")
    print("=" * 60)
    
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests_with_report()
    sys.exit(0 if success else 1)
