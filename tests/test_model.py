"""Tests for model components."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from adaptive_retrieval_qa_with_answerability_calibration.models.model import (
    AdaptiveRetrievalQAModel,
    ConfidenceCalibrator
)
from adaptive_retrieval_qa_with_answerability_calibration.utils.config import Config


class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator module."""

    def test_init(self):
        """Test ConfidenceCalibrator initialization."""
        calibrator = ConfidenceCalibrator(
            qa_hidden_size=768,
            confidence_features_dim=10,
            hidden_dim=256
        )

        assert calibrator.qa_hidden_size == 768
        assert calibrator.confidence_features_dim == 10
        assert hasattr(calibrator, 'input_projection')
        assert hasattr(calibrator, 'calibration_layers')
        assert hasattr(calibrator, 'temperature')

    def test_forward(self):
        """Test ConfidenceCalibrator forward pass."""
        calibrator = ConfidenceCalibrator(
            qa_hidden_size=768,
            confidence_features_dim=10
        )

        batch_size = 4
        seq_len = 128

        # Create dummy inputs
        qa_hidden_states = torch.randn(batch_size, 768)
        retrieval_scores = torch.randn(batch_size)
        confidence_features = torch.randn(batch_size, 10)
        start_logits = torch.randn(batch_size, seq_len)
        end_logits = torch.randn(batch_size, seq_len)

        # Forward pass
        answerability_logits, calibrated_confidence = calibrator(
            qa_hidden_states=qa_hidden_states,
            retrieval_scores=retrieval_scores,
            confidence_features=confidence_features,
            start_logits=start_logits,
            end_logits=end_logits
        )

        # Check outputs
        assert answerability_logits.shape == (batch_size,)
        assert calibrated_confidence.shape == (batch_size,)
        assert torch.all(calibrated_confidence >= 0) and torch.all(calibrated_confidence <= 1)

    def test_temperature_parameter(self):
        """Test temperature parameter behavior."""
        calibrator = ConfidenceCalibrator(qa_hidden_size=768, confidence_features_dim=10)

        # Check initial temperature
        assert calibrator.temperature.item() == 1.0

        # Test temperature update
        calibrator.temperature.data.fill_(2.0)
        assert calibrator.temperature.item() == 2.0


class TestAdaptiveRetrievalQAModel:
    """Tests for AdaptiveRetrievalQAModel."""

    def test_init(self, test_config: Config):
        """Test model initialization."""
        model = AdaptiveRetrievalQAModel(test_config)

        assert hasattr(model, 'qa_model')
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'retriever')
        assert hasattr(model, 'calibrator')
        assert model.confidence_threshold == test_config.get('model.confidence_threshold')
        assert model.retrieval_top_k == test_config.get('model.retrieval_top_k')

    def test_forward_basic(self, model: AdaptiveRetrievalQAModel, dummy_batch):
        """Test basic forward pass."""
        model.eval()

        with torch.no_grad():
            outputs = model(**dummy_batch)

        # Check required outputs
        assert 'start_logits' in outputs
        assert 'end_logits' in outputs
        assert 'answerability_logits' in outputs
        assert 'answerability_probs' in outputs

        # Check shapes
        batch_size = dummy_batch['input_ids'].size(0)
        seq_len = dummy_batch['input_ids'].size(1)

        assert outputs['start_logits'].shape == (batch_size, seq_len)
        assert outputs['end_logits'].shape == (batch_size, seq_len)
        assert outputs['answerability_logits'].shape == (batch_size,)
        assert outputs['answerability_probs'].shape == (batch_size,)

    def test_forward_with_labels(self, model: AdaptiveRetrievalQAModel, dummy_batch):
        """Test forward pass with labels (training mode)."""
        model.train()

        outputs = model(**dummy_batch)

        # Should have losses when labels are provided
        assert 'loss' in outputs
        assert 'qa_loss' in outputs
        assert 'calibration_loss' in outputs

        # Check that losses are scalars
        assert outputs['loss'].dim() == 0
        assert outputs['calibration_loss'].dim() == 0

    def test_forward_missing_features(self, model: AdaptiveRetrievalQAModel):
        """Test forward pass with missing optional features."""
        batch_size = 2
        seq_len = 64

        # Minimal inputs
        minimal_batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len)
        }

        with torch.no_grad():
            outputs = model(**minimal_batch)

        # Should still work with default values
        assert 'answerability_probs' in outputs
        assert outputs['answerability_probs'].shape == (batch_size,)

    def test_build_passage_index(self, model: AdaptiveRetrievalQAModel):
        """Test passage index building."""
        passages = [
            "The capital of France is Paris.",
            "London is the capital of England.",
            "Berlin is the capital of Germany."
        ]

        model.build_passage_index(passages)

        assert model.passage_corpus == passages
        assert model.passage_index is not None
        assert model.passage_embeddings is not None
        assert model.passage_embeddings.shape[0] == len(passages)

    def test_retrieve_passages(self, model: AdaptiveRetrievalQAModel):
        """Test passage retrieval."""
        passages = [
            "The capital of France is Paris.",
            "London is the capital of England.",
            "Berlin is the capital of Germany.",
            "Tokyo is the capital of Japan.",
            "Rome is the capital of Italy."
        ]

        model.build_passage_index(passages)

        query = "What is the capital of France?"
        retrieved_passages, scores = model.retrieve_passages(query, top_k=3)

        assert len(retrieved_passages) == 3
        assert len(scores) == 3
        assert all(isinstance(passage, str) for passage in retrieved_passages)
        assert all(isinstance(score, float) for score in scores)

    def test_retrieve_passages_without_index(self, model: AdaptiveRetrievalQAModel):
        """Test retrieval without building index first."""
        query = "What is the capital of France?"

        with pytest.raises(RuntimeError):
            model.retrieve_passages(query)

    def test_predict_basic(self, model: AdaptiveRetrievalQAModel):
        """Test basic prediction functionality."""
        # Build a simple passage index
        passages = [
            "Paris is the capital of France.",
            "London is the capital of England."
        ]
        model.build_passage_index(passages)

        question = "What is the capital of France?"
        result = model.predict(question)

        # Check result structure
        assert 'answer' in result
        assert 'confidence' in result
        assert 'is_answerable' in result
        assert 'retrieved_passages' in result
        assert 'retrieval_scores' in result

        # Check types
        assert isinstance(result['answer'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['is_answerable'], bool)
        assert isinstance(result['retrieved_passages'], list)
        assert isinstance(result['retrieval_scores'], list)

    def test_predict_with_contexts(self, model: AdaptiveRetrievalQAModel):
        """Test prediction with provided contexts."""
        question = "What is AI?"
        contexts = [
            "AI stands for artificial intelligence.",
            "Machine learning is a subset of AI."
        ]

        result = model.predict(question, contexts=contexts)

        assert result['retrieved_passages'] == contexts
        assert len(result['retrieval_scores']) == len(contexts)

    def test_predict_empty_contexts(self, model: AdaptiveRetrievalQAModel):
        """Test prediction with empty contexts."""
        question = "What is AI?"
        result = model.predict(question, contexts=[])

        assert result['answer'] == ''
        assert result['confidence'] == 0.0
        assert result['is_answerable'] is False

    @patch('torch.save')
    @patch('faiss.write_index')
    def test_save_pretrained(self, mock_faiss_write, mock_torch_save, model: AdaptiveRetrievalQAModel):
        """Test model saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "test_model"

            # Build passage index for testing
            passages = ["Test passage 1", "Test passage 2"]
            model.build_passage_index(passages)

            model.save_pretrained(str(save_dir))

            # Check that directories were created
            assert (save_dir / "qa_model").exists()
            assert (save_dir / "retriever").exists()

            # Check that save methods were called
            mock_torch_save.assert_called()

    def test_device_movement(self, model: AdaptiveRetrievalQAModel):
        """Test moving model between devices."""
        # Test CPU
        model.to('cpu')
        assert next(model.parameters()).device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            model.to('cuda')
            assert next(model.parameters()).device.type == 'cuda'
            model.to('cpu')  # Move back for other tests

    def test_training_mode_switching(self, model: AdaptiveRetrievalQAModel):
        """Test switching between training and evaluation modes."""
        # Test training mode
        model.train()
        assert model.training
        assert model.qa_model.training
        assert model.calibrator.training

        # Test evaluation mode
        model.eval()
        assert not model.training
        assert not model.qa_model.training
        assert not model.calibrator.training

    def test_gradient_computation(self, model: AdaptiveRetrievalQAModel, dummy_batch):
        """Test that gradients are computed in training mode."""
        model.train()

        # Forward pass
        outputs = model(**dummy_batch)
        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

        # Clear gradients
        model.zero_grad()

        # Check that gradients are cleared
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is None or torch.all(param.grad == 0)


class TestModelIntegration:
    """Integration tests for the complete model."""

    def test_end_to_end_training_step(self, model: AdaptiveRetrievalQAModel, dummy_batch):
        """Test a complete training step."""
        model.train()

        # Forward pass
        outputs = model(**dummy_batch)
        loss = outputs['loss']

        # Check loss properties
        assert loss.requires_grad
        assert loss.dim() == 0
        assert loss.item() >= 0

        # Backward pass
        loss.backward()

        # Check gradients
        grad_norms = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        assert len(grad_norms) > 0
        assert all(norm >= 0 for norm in grad_norms)

    def test_inference_pipeline(self, model: AdaptiveRetrievalQAModel):
        """Test complete inference pipeline."""
        # Setup
        passages = [
            "The capital of France is Paris, a major European city.",
            "London is the capital and largest city of England and the UK.",
            "Berlin is the capital and largest city of Germany."
        ]
        model.build_passage_index(passages)

        # Inference
        model.eval()
        question = "What is the capital of France?"

        with torch.no_grad():
            result = model.predict(question, return_confidence=True)

        # Verify complete result
        assert isinstance(result['answer'], str)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['is_answerable'], bool)
        assert 'passage_answerability_scores' in result
        assert 'best_passage_idx' in result

    def test_model_consistency(self, model: AdaptiveRetrievalQAModel):
        """Test model prediction consistency."""
        # Build passages
        passages = ["AI is artificial intelligence.", "ML is machine learning."]
        model.build_passage_index(passages)

        model.eval()
        question = "What is AI?"

        # Multiple predictions should be consistent
        with torch.no_grad():
            result1 = model.predict(question)
            result2 = model.predict(question)

        # Results should be identical (due to deterministic mode)
        assert result1['answer'] == result2['answer']
        assert abs(result1['confidence'] - result2['confidence']) < 1e-6

    def test_batch_processing(self, model: AdaptiveRetrievalQAModel):
        """Test processing multiple examples in a batch."""
        batch_size = 3
        seq_len = 128

        # Create batch
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'is_answerable': torch.tensor([1.0, 0.0, 1.0]),
            'retrieval_scores': torch.tensor([0.9, 0.5, 0.8]),
            'confidence_features': torch.randn(batch_size, 10)
        }

        model.eval()
        with torch.no_grad():
            outputs = model(**batch)

        # Check batch processing
        assert outputs['answerability_probs'].shape == (batch_size,)
        assert outputs['start_logits'].shape == (batch_size, seq_len)
        assert outputs['end_logits'].shape == (batch_size, seq_len)

    def test_model_memory_efficiency(self, model: AdaptiveRetrievalQAModel):
        """Test model memory usage is reasonable."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have reasonable number of parameters (not too many)
        assert total_params > 1000  # Has some parameters
        assert total_params < 1e9   # Not too large
        assert trainable_params <= total_params

        # Test memory usage with a batch
        dummy_input = {
            'input_ids': torch.randint(0, 1000, (8, 256)),  # Realistic batch
            'attention_mask': torch.ones(8, 256),
            'is_answerable': torch.ones(8),
            'retrieval_scores': torch.ones(8),
            'confidence_features': torch.randn(8, 10)
        }

        model.eval()
        with torch.no_grad():
            _ = model(**dummy_input)

        # Should complete without memory errors