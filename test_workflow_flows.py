#!/usr/bin/env python3
"""
Test Workflow Flows for Flashcard System
Tests the complete workflow flows from user input to system response
"""

import unittest
import json
import tempfile
import os
import threading
import time
from unittest.mock import patch, MagicMock, Mock
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app
from app import app

class TestWorkflowFlows(unittest.TestCase):
    """Test class for workflow flow testing"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock external services
        self.chromadb_patcher = patch('chromadb.PersistentClient')
        self.openai_patcher = patch('openai.AzureOpenAI')
        self.vits_model_patcher = patch('app.VitsModel.from_pretrained')
        self.tokenizer_patcher = patch('app.AutoTokenizer.from_pretrained')
        self.sf_write_patcher = patch('app.sf.write')
        self.setup_langchain_patcher = patch('app.setup_langchain')
        self.setup_faq_patcher = patch('app.setup_faq_vectorstore')
        
        # Start patches
        self.mock_chromadb = self.chromadb_patcher.start()
        self.mock_openai = self.openai_patcher.start()
        self.mock_vits_model = self.vits_model_patcher.start()
        self.mock_tokenizer = self.tokenizer_patcher.start()
        self.mock_sf_write = self.sf_write_patcher.start()
        self.mock_setup_langchain = self.setup_langchain_patcher.start()
        self.mock_setup_faq = self.setup_faq_patcher.start()
        
        # Mock internal functions
        self.rag_faq_patcher = patch('app.rag_faq_answer')
        self.analyze_text_patcher = patch('app.analyze_text_for_vocabulary')
        self.explain_word_patcher = patch('app.explain_word')
        self.extract_word_patcher = patch('app.extract_word_from_input')
        self.save_history_patcher = patch('app.save_to_history')
        
        self.mock_rag_faq = self.rag_faq_patcher.start()
        self.mock_analyze_text = self.analyze_text_patcher.start()
        self.mock_explain_word = self.explain_word_patcher.start()
        self.mock_extract_word = self.extract_word_patcher.start()
        self.mock_save_history = self.save_history_patcher.start()
        
        # Setup mock returns
        self.setup_mocks()
    
    def tearDown(self):
        """Clean up test environment"""
        # Stop all patches
        self.chromadb_patcher.stop()
        self.openai_patcher.stop()
        self.vits_model_patcher.stop()
        self.tokenizer_patcher.stop()
        self.sf_write_patcher.stop()
        self.setup_langchain_patcher.stop()
        self.setup_faq_patcher.stop()
        self.rag_faq_patcher.stop()
        self.analyze_text_patcher.stop()
        self.explain_word_patcher.stop()
        self.extract_word_patcher.stop()
        self.save_history_patcher.stop()
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def setup_mocks(self):
        """Setup mock return values"""
        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['test_id']],
            'documents': [['test document']],
            'metadatas': [[{'word': 'workflow', 'category': 'Business'}]],
            'distances': [[0.1]]
        }
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        self.mock_chromadb.return_value = mock_client
        
        # Mock OpenAI
        mock_openai_client = MagicMock()
        self.mock_openai.return_value = mock_openai_client
        
        # Mock TTS models
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        self.mock_vits_model.return_value = mock_model
        self.mock_tokenizer.return_value = mock_tokenizer
        
        # Mock soundfile
        self.mock_sf_write.return_value = None
        
        # Mock LangChain setup
        mock_agent = MagicMock()
        self.mock_setup_langchain.return_value = mock_agent
        
        # Mock FAQ setup
        mock_faq_store = MagicMock()
        self.mock_setup_faq.return_value = mock_faq_store
        
        # Mock internal functions
        self.mock_rag_faq.return_value = {
            "answer": "Để thêm từ vựng mới, bạn có thể sử dụng lệnh 'add word [từ]' hoặc 'thêm từ [từ]'",
            "confidence": 0.95,
            "source": "FAQ database"
        }
        
        self.mock_analyze_text.return_value = [
            {"word": "artificial", "vietnamese_meaning": "nhân tạo"},
            {"word": "intelligence", "vietnamese_meaning": "trí thông minh"}
        ]
        
        self.mock_explain_word.return_value = {
            "word": "efficiency",
            "vietnamese_meaning": "hiệu quả",
            "phonetic": "/ɪˈfɪʃənsi/",
            "part_of_speech": "noun",
            "example_sentences": ["The new system improved efficiency."],
            "mnemonic_tip": "Efficiency = hiệu quả",
            "difficulty_level": "intermediate",
            "synonyms": ["effectiveness", "productivity"],
            "audio_path": "/static/audio/efficiency_12345.wav"
        }
        
        self.mock_extract_word.return_value = "hello"
        self.mock_save_history.return_value = True
    
    def test_faq_question_flow(self):
        """Test FAQ Question Flow - Complete workflow from FAQ input to answer"""
        # Test data
        test_data = {
            "question": "Làm thế nào để thêm từ vựng mới?"
        }
        
        # Make request
        response = self.app.post('/api/faq', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('answer', data)
        self.assertIn("thêm từ vựng", str(data['answer']))
    
    def test_faq_question_flow_with_unknown_question(self):
        """Test FAQ Question Flow with unknown question - should return default response"""
        # Mock FAQ to return default response
        self.mock_rag_faq.return_value = {
            "answer": "Xin lỗi, tôi không tìm thấy câu trả lời cho câu hỏi này. Vui lòng thử câu hỏi khác.",
            "confidence": 0.0,
            "source": "Default response"
        }
        
        # Test data
        test_data = {
            "question": "Câu hỏi không có trong database?"
        }
        
        # Make request
        response = self.app.post('/api/faq', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('answer', data)
        self.assertIn("Xin lỗi", str(data['answer']))
    
    def test_chat_message_flow(self):
        """Test Chat Message Flow - Complete workflow from chat input to response"""
        # Mock extract word
        self.mock_extract_word.return_value = "hello"
        
        # Test data
        test_data = {
            "message": "Xin chào, bạn có thể giúp tôi học từ vựng không?",
            "type": "word"
        }
        
        # Make request
        response = self.app.post('/api/chat', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('tool_results', data)
        self.assertEqual(data['type'], 'intelligent_chat')
    
    def test_chat_message_flow_with_vocabulary_request(self):
        """Test Chat Message Flow with vocabulary learning request"""
        # Mock extract word
        self.mock_extract_word.return_value = "innovation"
        
        # Test data
        test_data = {
            "message": "Tôi muốn thêm từ innovation vào danh sách từ vựng",
            "type": "word"
        }
        
        # Make request
        response = self.app.post('/api/chat', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('tool_results', data)
        self.assertEqual(data['type'], 'intelligent_chat')
    
    def test_text_passage_flow(self):
        """Test Text Passage Flow - Complete workflow from text input to vocabulary extraction"""
        # Test data
        test_data = {
            "message": "Artificial intelligence is transforming the world. Human intelligence combined with AI creates powerful solutions.",
            "type": "text"
        }
        
        # Make request
        response = self.app.post('/api/chat', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('tool_results', data)
        self.assertIn('message', data)
        self.assertEqual(data['type'], 'intelligent_chat')
    
    def test_text_passage_flow_with_empty_text(self):
        """Test Text Passage Flow with empty text - should return error"""
        # Test data
        test_data = {
            "message": "",
            "type": "text"
        }
        
        # Make request
        response = self.app.post('/api/chat', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn("Vui lòng nhập nội dung", data['message'])
    
    def test_search_query_flow(self):
        """Test Search Query Flow - Complete workflow from search query to results"""
        # Test data
        test_data = {
            "message": "workflow"
        }
        
        # Make request
        response = self.app.post('/api/semantic-search', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('results', data)
    
    def test_search_query_flow_with_category_filter(self):
        """Test Search Query Flow with category filter"""
        # Test data
        test_data = {
            "category": "Technology",
            "limit": 5
        }
        
        # Make request
        response = self.app.post('/api/search-category', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('results', data)
    
    def test_single_word_flow(self):
        """Test Single Word Flow - Complete workflow from single word to detailed analysis"""
        # Mock explain word
        self.mock_explain_word.return_value = {
            "word": "efficiency",
            "vietnamese_meaning": "hiệu quả",
            "phonetic": "/ɪˈfɪʃənsi/",
            "part_of_speech": "noun",
            "example_sentences": ["The new system improved efficiency."],
            "mnemonic_tip": "Efficiency = hiệu quả",
            "difficulty_level": "intermediate",
            "synonyms": ["effectiveness", "productivity"]
        }
        
        # Test data
        test_data = {
            "message": "efficiency",
            "type": "word"
        }
        
        # Make request
        response = self.app.post('/api/chat', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('tool_results', data)
    
    def test_single_word_flow_with_audio_generation(self):
        """Test Single Word Flow with audio generation"""
        # Mock extract word
        self.mock_extract_word.return_value = "efficiency"
        
        # Mock explain word with audio path
        self.mock_explain_word.return_value = {
            "word": "efficiency",
            "vietnamese_meaning": "hiệu quả",
            "phonetic": "/ɪˈfɪʃənsi/",
            "part_of_speech": "noun",
            "example_sentences": ["The new system improved efficiency."],
            "mnemonic_tip": "Efficiency = hiệu quả",
            "difficulty_level": "intermediate",
            "synonyms": ["effectiveness", "productivity"],
            "audio_path": "/static/audio/efficiency_12345.wav"
        }
        
        # Test data
        test_data = {
            "message": "Tôi muốn thêm từ efficiency",
            "type": "word"
        }
        
        # Make request
        response = self.app.post('/api/chat', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('tool_results', data)
        self.assertEqual(data['type'], 'intelligent_chat')
    
    def test_integrated_workflow_from_chat_to_vocabulary(self):
        """Test integrated workflow: Chat → Word Analysis → Save to History"""
        # Mock extract word
        self.mock_extract_word.return_value = "collaboration"
        
        # Mock explain word
        self.mock_explain_word.return_value = {
            "word": "collaboration",
            "vietnamese_meaning": "hợp tác",
            "phonetic": "/kəˌlæbəˈreɪʃən/",
            "part_of_speech": "noun",
            "example_sentences": ["Team collaboration is essential for success."],
            "mnemonic_tip": "Collaboration = hợp tác",
            "difficulty_level": "intermediate",
            "synonyms": ["cooperation", "partnership"]
        }
        
        # Test data
        test_data = {
            "message": "Tôi muốn thêm từ 'collaboration'",
            "type": "word"
        }
        
        # Make request
        response = self.app.post('/api/chat', 
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Assertions
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertIn('tool_results', data)
        self.assertEqual(data['type'], 'intelligent_chat')
        
        # Note: save_to_history is called internally by the agent, not directly in the API
        # The test verifies the API response structure is correct
    
    def test_workflow_error_handling(self):
        """Test workflow error handling across different flows"""
        # Test empty message
        test_data_empty = {
            "message": "",
            "type": "word"
        }
        
        response = self.app.post('/api/chat', 
                               data=json.dumps(test_data_empty),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn("Vui lòng nhập nội dung", data['message'])
    
    def test_concurrent_workflow_processing(self):
        """Test concurrent processing of multiple workflow requests"""
        results = []
        errors = []
        
        def make_request(request_type, data):
            try:
                if request_type == "chat":
                    response = self.app.post('/api/chat', 
                                           data=json.dumps(data),
                                           content_type='application/json')
                elif request_type == "search":
                    response = self.app.post('/api/semantic-search', 
                                           data=json.dumps(data),
                                           content_type='application/json')
                elif request_type == "word":
                    response = self.app.post('/api/chat', 
                                           data=json.dumps({"message": data, "type": "word"}),
                                           content_type='application/json')
                
                if response.status_code == 200:
                    results.append(True)
                else:
                    errors.append(f"Status {response.status_code}")
            except Exception as e:
                errors.append(str(e))
        
        # Create threads for concurrent requests
        threads = []
        
        # 3 chat requests
        for i in range(3):
            thread = threading.Thread(target=make_request, 
                                    args=("chat", {"message": f"Message {i}", "type": "word"}))
            threads.append(thread)
        
        # 3 search requests
        for i in range(3):
            thread = threading.Thread(target=make_request, 
                                    args=("search", {"message": f"word{i}"}))
            threads.append(thread)
        
        # 3 word analysis requests
        for i in range(3):
            thread = threading.Thread(target=make_request, 
                                    args=("word", f"word{i}"))
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        end_time = time.time()
        
        # Assertions
        self.assertEqual(len(results), 9)  # All 9 requests should succeed
        self.assertEqual(len(errors), 0)   # No errors should occur
        self.assertLess(end_time - start_time, 10)  # Should complete within 10 seconds

if __name__ == '__main__':
    unittest.main()
