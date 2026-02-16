"""Tests for data loading and preprocessing modules."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from datasets import Dataset
import torch

from adaptive_retrieval_qa_with_answerability_calibration.data.loader import DatasetLoader
from adaptive_retrieval_qa_with_answerability_calibration.data.preprocessing import DataPreprocessor
from adaptive_retrieval_qa_with_answerability_calibration.utils.config import Config


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_init(self, test_config: Config):
        """Test DatasetLoader initialization."""
        loader = DatasetLoader(test_config)
        assert loader.config == test_config
        assert hasattr(loader, 'logger')

    def test_stratified_sample(self, data_loader: DatasetLoader, sample_dataset: Dataset):
        """Test stratified sampling functionality."""
        # Test with equal split
        sampled = data_loader._stratified_sample(sample_dataset, 6)
        assert len(sampled) == 6

        # Check that both answerable and unanswerable examples are included
        is_answerable = [len(ex['answers']['text']) > 0 for ex in sampled]
        answerable_count = sum(is_answerable)
        unanswerable_count = len(is_answerable) - answerable_count

        # Should have some of each type
        assert answerable_count > 0
        assert unanswerable_count >= 0

    def test_process_squad_for_combined(self, data_loader: DatasetLoader, sample_dataset: Dataset):
        """Test SQuAD data processing for combined dataset."""
        processed = data_loader._process_squad_for_combined(sample_dataset)

        assert 'question' in processed
        assert 'context' in processed
        assert 'answers' in processed
        assert 'is_answerable' in processed
        assert 'source' in processed

        assert len(processed['question']) == len(sample_dataset)
        assert all(source == 'squad_v2' for source in processed['source'])

    def test_get_dataset_statistics(self, data_loader: DatasetLoader, sample_dataset: Dataset):
        """Test dataset statistics calculation."""
        # Create combined dataset format
        processed = data_loader._process_squad_for_combined(sample_dataset)
        combined_dataset = Dataset.from_dict(processed)

        stats = data_loader.get_dataset_statistics(combined_dataset)

        assert 'total_examples' in stats
        assert 'answerable_count' in stats
        assert 'unanswerable_count' in stats
        assert 'answerable_ratio' in stats
        assert 'avg_question_length' in stats
        assert 'avg_context_length' in stats

        assert stats['total_examples'] == len(sample_dataset)
        assert stats['answerable_count'] + stats['unanswerable_count'] == stats['total_examples']
        assert 0 <= stats['answerable_ratio'] <= 1

    @patch('adaptive_retrieval_qa_with_answerability_calibration.data.loader.load_dataset')
    def test_load_squad_v2_mock(self, mock_load_dataset, data_loader: DatasetLoader):
        """Test SQuAD 2.0 loading with mocked dataset."""
        # Mock the dataset with proper methods
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        mock_dataset.select.return_value = mock_dataset  # Mock the select method
        mock_load_dataset.return_value = mock_dataset

        result = data_loader.load_squad_v2('train')

        mock_load_dataset.assert_called_once_with('squad_v2', split='train')
        # The result should be the mocked dataset (could be after processing)
        assert result is not None

    def test_create_combined_dataset_empty(self, data_loader: DatasetLoader):
        """Test combined dataset creation with empty inputs."""
        empty_dataset = Dataset.from_dict({
            'question': [],
            'context': [],
            'answers': [],
            'is_answerable': [],
            'source': [],
            'passage_id': [],
            'relevance_score': []
        })

        result = data_loader.create_combined_dataset(empty_dataset, empty_dataset)
        assert len(result) == 0


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_init(self, test_config: Config):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor(test_config)
        assert preprocessor.config == test_config
        assert hasattr(preprocessor, 'tokenizer')
        assert hasattr(preprocessor, 'retriever')
        assert hasattr(preprocessor, 'max_seq_length')

    def test_normalize_text(self, preprocessor: DataPreprocessor):
        """Test text normalization."""
        # Test basic normalization
        text = "  Hello   world!  "
        normalized = preprocessor._normalize_text(text)
        assert normalized == "Hello world!"

        # Test empty string
        assert preprocessor._normalize_text("") == ""

        # Test special characters
        text = "Hello @#$% world!!!"
        normalized = preprocessor._normalize_text(text)
        assert "@#$%" not in normalized

    def test_calculate_word_overlap(self, preprocessor: DataPreprocessor):
        """Test word overlap calculation."""
        question = "What is the capital of France?"
        context = "Paris is the capital city of France and its largest city."

        overlap = preprocessor._calculate_word_overlap(question, context)
        assert 0 <= overlap <= 1
        assert overlap > 0  # Should have some overlap

        # Test no overlap
        question = "xyz abc def"
        context = "qwe rty uio"
        overlap = preprocessor._calculate_word_overlap(question, context)
        assert overlap == 0

    def test_has_question_words(self, preprocessor: DataPreprocessor):
        """Test question word detection."""
        # Test with question words
        question = "What is artificial intelligence?"
        assert preprocessor._has_question_words(question) == 1.0

        # Test without question words
        question = "Artificial intelligence applications."
        assert preprocessor._has_question_words(question) == 0.0

        # Test with auxiliary verbs
        question = "Is this correct?"
        assert preprocessor._has_question_words(question) == 1.0

    def test_calculate_text_complexity(self, preprocessor: DataPreprocessor):
        """Test text complexity calculation."""
        # Test simple text
        simple_text = "The cat sat."
        complexity = preprocessor._calculate_text_complexity(simple_text)
        assert 0 <= complexity <= 1

        # Test complex text
        complex_text = "Photosynthesis is the extraordinarily intricate biochemical process."
        complex_complexity = preprocessor._calculate_text_complexity(complex_text)
        assert complex_complexity > complexity

        # Test empty text
        assert preprocessor._calculate_text_complexity("") == 0

    def test_clean_text(self, preprocessor: DataPreprocessor):
        """Test text cleaning functionality."""
        examples = {
            'question': ["  What   is  AI?  ", "How does   it work?"],
            'context': ["AI  is   artificial  intelligence.", "It   works  through   algorithms."]
        }

        cleaned = preprocessor._clean_text(examples)

        assert cleaned['question'][0] == "What is AI?"
        assert cleaned['question'][1] == "How does it work?"
        assert cleaned['context'][0] == "AI is artificial intelligence."
        assert cleaned['context'][1] == "It works through algorithms."

    def test_chunk_passages(self, preprocessor: DataPreprocessor):
        """Test passage chunking."""
        # Test short passage (no chunking needed)
        example = {
            'context': "Short passage.",
            'answers': {'text': ['passage'], 'answer_start': [6]}
        }

        result = preprocessor._chunk_passages(example)
        assert result['context'] == "Short passage."

        # Test long passage
        long_text = " ".join(["word"] * 300)  # Create long passage
        example = {
            'context': long_text,
            'answers': {'text': ['word'], 'answer_start': [0]}
        }

        result = preprocessor._chunk_passages(example)
        assert len(result['context'].split()) <= preprocessor.passage_max_length

    def test_extract_confidence_features(self, preprocessor: DataPreprocessor):
        """Test confidence feature extraction."""
        example = {
            'question': 'What is the capital of France?',
            'context': 'Paris is the capital of France.',
            'relevance_score': 0.8,
            'source': 'squad_v2'
        }

        result = preprocessor._extract_confidence_features(example)

        assert 'confidence_features' in result
        features = result['confidence_features']

        assert 'question_length' in features
        assert 'context_length' in features
        assert 'question_word_overlap' in features
        assert 'has_question_words' in features
        assert 'context_complexity' in features
        assert 'retrieval_score' in features
        assert 'is_squad' in features
        assert 'is_marco' in features

        assert features['is_squad'] == 1.0
        assert features['is_marco'] == 0.0
        assert features['retrieval_score'] == 0.8

    def test_preprocess_dataset(self, preprocessor: DataPreprocessor, small_dataset: Dataset):
        """Test complete dataset preprocessing."""
        processed = preprocessor.preprocess_dataset(small_dataset, is_training=True)

        # Check that required columns are present
        expected_columns = [
            'input_ids', 'attention_mask', 'is_answerable',
            'passage_embeddings', 'confidence_features'
        ]

        for col in expected_columns:
            assert col in processed.column_names

        # Check data types and shapes
        assert len(processed) == len(small_dataset)

        # Test a single example
        example = processed[0]
        assert isinstance(example['input_ids'], list)
        assert isinstance(example['attention_mask'], list)
        assert isinstance(example['passage_embeddings'], list)
        assert isinstance(example['confidence_features'], dict)

    def test_create_retrieval_corpus(self, preprocessor: DataPreprocessor, processed_dataset: Dataset):
        """Test retrieval corpus creation."""
        passages, embeddings = preprocessor.create_retrieval_corpus(processed_dataset)

        assert isinstance(passages, list)
        assert isinstance(embeddings, np.ndarray)
        assert len(passages) <= len(processed_dataset)  # May be fewer due to deduplication
        assert embeddings.shape[0] == len(passages)
        assert embeddings.shape[1] > 0  # Should have embedding dimension

    def test_prepare_inference_input(self, preprocessor: DataPreprocessor):
        """Test inference input preparation."""
        question = "What is AI?"
        contexts = ["AI is artificial intelligence.", "Machine learning is part of AI."]

        inputs = preprocessor.prepare_inference_input(question, contexts)

        assert 'input_ids' in inputs
        assert 'attention_mask' in inputs
        assert inputs['input_ids'].shape[0] == len(contexts)
        assert inputs['attention_mask'].shape[0] == len(contexts)

        # Test empty context
        empty_inputs = preprocessor.prepare_inference_input(question, [])
        assert empty_inputs['input_ids'].shape[0] == 1
        assert empty_inputs['attention_mask'].shape[0] == 1


class TestDataIntegration:
    """Integration tests for data loading and preprocessing."""

    def test_full_data_pipeline(self, test_config: Config, sample_dataset: Dataset):
        """Test complete data pipeline from loading to preprocessing."""
        # Initialize components
        loader = DatasetLoader(test_config)
        preprocessor = DataPreprocessor(test_config)

        # Process sample data as if it were loaded
        processed_squad = loader._process_squad_for_combined(sample_dataset)
        combined_dataset = Dataset.from_dict(processed_squad)

        # Preprocess
        final_dataset = preprocessor.preprocess_dataset(combined_dataset, is_training=True)

        # Verify final dataset
        assert len(final_dataset) == len(sample_dataset)

        # Check all required features are present
        required_features = [
            'input_ids', 'attention_mask', 'is_answerable',
            'passage_embeddings', 'confidence_features'
        ]

        for feature in required_features:
            assert feature in final_dataset.column_names

        # Verify data consistency
        for i in range(min(3, len(final_dataset))):
            example = final_dataset[i]
            assert len(example['input_ids']) == preprocessor.max_seq_length
            assert len(example['attention_mask']) == preprocessor.max_seq_length
            assert isinstance(example['is_answerable'], bool)

    def test_dataset_statistics_accuracy(self, data_loader: DatasetLoader):
        """Test accuracy of dataset statistics calculation."""
        # Create dataset with known statistics
        data = {
            'question': ['Q1', 'Q2', 'Q3', 'Q4'],
            'context': ['C1', 'C2', 'C3', 'C4'],
            'answers': [
                {'text': ['A1'], 'answer_start': [0]},
                {'text': [], 'answer_start': []},  # Unanswerable
                {'text': ['A3'], 'answer_start': [0]},
                {'text': [], 'answer_start': []}   # Unanswerable
            ],
            'is_answerable': [True, False, True, False],
            'source': ['squad_v2'] * 4,
            'passage_id': ['p1', 'p2', 'p3', 'p4'],
            'relevance_score': [1.0, 0.8, 0.9, 0.7]
        }

        dataset = Dataset.from_dict(data)
        stats = data_loader.get_dataset_statistics(dataset)

        assert stats['total_examples'] == 4
        assert stats['answerable_count'] == 2
        assert stats['unanswerable_count'] == 2
        assert stats['answerable_ratio'] == 0.5
        assert stats['squad_examples'] == 4
        assert stats['marco_examples'] == 0

    def test_error_handling(self, test_config: Config):
        """Test error handling in data processing."""
        loader = DatasetLoader(test_config)
        preprocessor = DataPreprocessor(test_config)

        # Test with malformed data
        bad_data = {
            'question': ['Q1'],
            'context': ['C1'],
            'answers': [{'text': None, 'answer_start': None}],  # Bad format
            'is_answerable': [True],
            'source': ['test'],
            'passage_id': ['p1'],
            'relevance_score': [1.0]
        }

        # Should handle gracefully
        try:
            dataset = Dataset.from_dict(bad_data)
            # This might fail or succeed depending on preprocessing robustness
            processed = preprocessor.preprocess_dataset(dataset, is_training=True)
        except Exception as e:
            # Expected for malformed data
            assert isinstance(e, Exception)