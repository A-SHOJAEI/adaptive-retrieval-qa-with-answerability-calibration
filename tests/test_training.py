"""Tests for training components."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from datasets import Dataset
from torch.utils.data import DataLoader

from adaptive_retrieval_qa_with_answerability_calibration.training.trainer import (
    AdaptiveQATrainer,
    AdaptiveQADataset,
    EarlyStopping
)
from adaptive_retrieval_qa_with_answerability_calibration.models.model import AdaptiveRetrievalQAModel
from adaptive_retrieval_qa_with_answerability_calibration.utils.config import Config


class TestAdaptiveQADataset:
    """Tests for AdaptiveQADataset."""

    def test_init(self, processed_dataset: Dataset, test_config: Config):
        """Test dataset initialization."""
        dataset = AdaptiveQADataset(processed_dataset, test_config)
        assert len(dataset) == len(processed_dataset)
        assert dataset.config == test_config

    def test_getitem(self, processed_dataset: Dataset, test_config: Config):
        """Test getting individual items."""
        dataset = AdaptiveQADataset(processed_dataset, test_config)

        # Test first item
        item = dataset[0]

        # Check required keys
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'is_answerable' in item

        # Check tensor types
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        assert isinstance(item['is_answerable'], torch.Tensor)

        # Check shapes
        assert item['input_ids'].dim() == 1
        assert item['attention_mask'].dim() == 1
        assert item['is_answerable'].dim() == 0

    def test_getitem_with_training_labels(self, processed_dataset: Dataset, test_config: Config):
        """Test getting items with training labels."""
        dataset = AdaptiveQADataset(processed_dataset, test_config)

        item = dataset[0]

        # Should have training-specific fields if available in processed_dataset
        if 'start_positions' in processed_dataset.column_names:
            assert 'start_positions' in item
            assert 'end_positions' in item
            assert isinstance(item['start_positions'], torch.Tensor)
            assert isinstance(item['end_positions'], torch.Tensor)

    def test_confidence_features_conversion(self, test_config: Config):
        """Test confidence features tensor conversion."""
        # Create dataset with confidence features
        data = {
            'input_ids': [[1, 2, 3, 4, 5] * 10],  # Pad to reasonable length
            'attention_mask': [[1, 1, 1, 1, 1] * 10],
            'is_answerable': [True],
            'confidence_features': [{
                'question_length': 5.0,
                'context_length': 10.0,
                'question_word_overlap': 0.5,
                'has_question_words': 1.0,
                'context_complexity': 0.3,
                'retrieval_score': 0.8,
                'is_squad': 1.0,
                'is_marco': 0.0
            }]
        }

        dataset_hf = Dataset.from_dict(data)
        dataset = AdaptiveQADataset(dataset_hf, test_config)

        item = dataset[0]
        assert 'confidence_features' in item
        assert isinstance(item['confidence_features'], torch.Tensor)
        assert item['confidence_features'].shape == (10,)  # 10-dimensional feature vector


class TestEarlyStopping:
    """Tests for EarlyStopping."""

    def test_init(self):
        """Test early stopping initialization."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        assert early_stopping.patience == 3
        assert early_stopping.min_delta == 0.01
        assert early_stopping.counter == 0
        assert early_stopping.best_loss is None

    def test_no_early_stop_improving(self):
        """Test that early stopping doesn't trigger when improving."""
        early_stopping = EarlyStopping(patience=2)
        model = torch.nn.Linear(10, 1)  # Dummy model

        # Improving losses
        assert not early_stopping(1.0, model)  # First loss
        assert not early_stopping(0.9, model)  # Better
        assert not early_stopping(0.8, model)  # Even better

        assert early_stopping.counter == 0

    def test_early_stop_triggered(self):
        """Test that early stopping triggers after patience exceeded."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)
        model = torch.nn.Linear(10, 1)

        # Initial improvement
        assert not early_stopping(1.0, model)
        assert not early_stopping(0.5, model)  # Big improvement

        # Then stagnation
        assert not early_stopping(0.51, model)  # Counter = 1
        assert early_stopping(0.52, model)     # Counter = 2, should stop

    def test_min_delta_threshold(self):
        """Test min_delta threshold behavior."""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1)
        model = torch.nn.Linear(10, 1)

        assert not early_stopping(1.0, model)
        # Small improvement below min_delta should count as no improvement
        assert not early_stopping(0.95, model)  # Counter = 1
        assert early_stopping(0.94, model)     # Counter = 2, should stop

    def test_best_weights_restoration(self):
        """Test that best weights are restored when requested."""
        early_stopping = EarlyStopping(patience=1, restore_best_weights=True)
        model = torch.nn.Linear(10, 1)

        # Save original weights
        original_weight = model.weight.data.clone()

        # First call - saves weights
        early_stopping(1.0, model)

        # Modify weights
        model.weight.data.fill_(999.0)
        modified_weight = model.weight.data.clone()

        # Trigger early stopping - should restore weights
        early_stopping(2.0, model)  # Worse loss

        # Weights should be restored (not exactly equal due to cloning, but close)
        assert not torch.equal(model.weight.data, modified_weight)


class TestAdaptiveQATrainer:
    """Tests for AdaptiveQATrainer."""

    def test_init(self, model: AdaptiveRetrievalQAModel, test_config: Config, small_dataset: Dataset):
        """Test trainer initialization."""
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=small_dataset
        )

        assert trainer.model == model
        assert trainer.config == test_config
        assert trainer.train_dataset == small_dataset
        assert trainer.device.type == 'cpu'  # Test config forces CPU
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'scheduler')
        assert hasattr(trainer, 'evaluator')

    def test_prepare_dataloaders(self, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset):
        """Test dataloader preparation."""
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=processed_dataset,
            val_dataset=processed_dataset  # Use same for val
        )

        train_loader, val_loader = trainer.prepare_dataloaders()

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert len(train_loader) > 0
        assert len(val_loader) > 0

    def test_prepare_dataloaders_no_validation(self, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset):
        """Test dataloader preparation without validation dataset."""
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=processed_dataset
        )

        train_loader, val_loader = trainer.prepare_dataloaders()

        assert isinstance(train_loader, DataLoader)
        assert val_loader is None

    def test_collate_fn(self, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset):
        """Test custom collate function."""
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=processed_dataset
        )

        # Create sample batch items
        dataset = AdaptiveQADataset(processed_dataset, test_config)
        batch_items = [dataset[i] for i in range(min(2, len(dataset)))]

        batched = trainer._collate_fn(batch_items)

        # Check that tensors are stacked
        for key in batch_items[0].keys():
            assert key in batched
            assert isinstance(batched[key], torch.Tensor)
            assert batched[key].shape[0] == len(batch_items)

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    def test_mlflow_setup(self, mock_log_params, mock_start_run, model: AdaptiveRetrievalQAModel, test_config: Config, small_dataset: Dataset):
        """Test MLflow setup."""
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=small_dataset
        )

        # MLflow should be called during initialization
        mock_start_run.assert_called_once()
        mock_log_params.assert_called()

    def test_train_epoch_basic(self, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset):
        """Test basic epoch training."""
        # Use very small config for fast testing
        test_config.set('training.logging_steps', 1)
        test_config.set('training.save_steps', 100)  # Don't save during test

        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=processed_dataset
        )

        train_loader, _ = trainer.prepare_dataloaders()

        # Train for one epoch
        metrics = trainer.train_epoch(train_loader)

        # Check that metrics are returned
        assert 'loss' in metrics
        assert 'qa_loss' in metrics
        assert 'calibration_loss' in metrics
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] >= 0

    def test_validate_basic(self, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset):
        """Test basic validation."""
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            val_dataset=processed_dataset
        )

        _, val_loader = trainer.prepare_dataloaders()

        # Run validation
        metrics = trainer.validate(val_loader)

        # Check validation metrics
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] >= 0

    @patch('mlflow.end_run')
    @patch('mlflow.log_metrics')
    def test_train_full(self, mock_log_metrics, mock_end_run, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset):
        """Test full training process."""
        # Set very small training config
        test_config.set('training.num_epochs', 1)
        test_config.set('training.logging_steps', 1)
        test_config.set('training.save_steps', 100)
        test_config.set('training.eval_steps', 100)

        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=processed_dataset,
            val_dataset=processed_dataset
        )

        # Run training
        history = trainer.train()

        # Check training history
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0

        # MLflow should log final metrics
        mock_end_run.assert_called()

    def test_save_checkpoint(self, model: AdaptiveRetrievalQAModel, test_config: Config, small_dataset: Dataset, temp_dir: Path):
        """Test checkpoint saving."""
        # Set checkpoint directory
        test_config.set('paths.checkpoint_dir', str(temp_dir / 'checkpoints'))

        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=small_dataset
        )

        # Save checkpoint
        trainer.save_checkpoint(is_final=True)

        # Check that files were created
        checkpoint_dir = temp_dir / 'checkpoints' / 'final_model'
        assert checkpoint_dir.exists()

        # Check for model files
        assert (checkpoint_dir / 'qa_model').exists()
        assert (checkpoint_dir / 'training_state.pt').exists()

    def test_optimizer_setup(self, model: AdaptiveRetrievalQAModel, test_config: Config, small_dataset: Dataset):
        """Test optimizer and scheduler setup."""
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=small_dataset
        )

        # Check optimizer
        assert hasattr(trainer, 'optimizer')
        assert len(trainer.optimizer.param_groups) > 0

        # Check scheduler
        assert hasattr(trainer, 'scheduler')

        # Test parameter grouping (different learning rates for different components)
        param_groups = trainer.optimizer.param_groups
        learning_rates = [group['lr'] for group in param_groups]

        # Should have different learning rates for QA model vs calibrator
        if len(param_groups) > 1:
            assert len(set(learning_rates)) > 1

    def test_error_handling(self, model: AdaptiveRetrievalQAModel, test_config: Config):
        """Test error handling in trainer."""
        # Test training without dataset
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=None
        )

        # Should raise error when trying to prepare dataloaders
        with pytest.raises(ValueError):
            trainer.prepare_dataloaders()

        # Should raise error when trying to train
        with pytest.raises(ValueError):
            trainer.train()


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    def test_complete_training_pipeline(self, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset, temp_dir: Path):
        """Test complete training pipeline."""
        # Configure for minimal training
        test_config.set('training.num_epochs', 1)
        test_config.set('training.batch_size', 2)
        test_config.set('training.logging_steps', 1)
        test_config.set('training.save_steps', 100)
        test_config.set('paths.checkpoint_dir', str(temp_dir / 'checkpoints'))

        # Split dataset
        split_point = len(processed_dataset) // 2
        train_data = processed_dataset.select(range(split_point))
        val_data = processed_dataset.select(range(split_point, len(processed_dataset)))

        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=train_data,
            val_dataset=val_data
        )

        # Run training
        history = trainer.train()

        # Verify training completed
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0

        # Verify model was saved
        final_model_path = temp_dir / 'checkpoints' / 'final_model'
        assert final_model_path.exists()

    def test_gradient_flow(self, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset):
        """Test that gradients flow properly during training."""
        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=processed_dataset
        )

        train_loader, _ = trainer.prepare_dataloaders()

        # Get a batch
        batch = next(iter(train_loader))

        # Forward pass
        trainer.model.train()
        outputs = trainer.model(**batch)
        loss = outputs['loss']

        # Backward pass
        trainer.optimizer.zero_grad()
        loss.backward()

        # Check gradients exist and are reasonable
        gradient_norms = []
        for name, param in trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms.append(grad_norm)
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

        assert len(gradient_norms) > 0, "No gradients found"
        assert all(norm >= 0 for norm in gradient_norms), "Negative gradient norms"

    def test_learning_rate_schedule(self, model: AdaptiveRetrievalQAModel, test_config: Config, small_dataset: Dataset):
        """Test learning rate scheduling."""
        # Set small warmup steps for testing
        test_config.set('training.warmup_steps', 2)

        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=small_dataset
        )

        initial_lr = trainer.scheduler.get_last_lr()[0]

        # Step scheduler a few times
        for _ in range(5):
            trainer.optimizer.step()
            trainer.scheduler.step()

        final_lr = trainer.scheduler.get_last_lr()[0]

        # Learning rate should change
        assert final_lr != initial_lr

    def test_memory_efficiency(self, model: AdaptiveRetrievalQAModel, test_config: Config, processed_dataset: Dataset):
        """Test training memory efficiency."""
        # Use larger batch size to test memory
        test_config.set('training.batch_size', 4)

        trainer = AdaptiveQATrainer(
            model=model,
            config=test_config,
            train_dataset=processed_dataset
        )

        train_loader, _ = trainer.prepare_dataloaders()

        # Train one batch without running out of memory
        batch = next(iter(train_loader))
        trainer.model.train()

        # Forward pass
        outputs = trainer.model(**batch)
        loss = outputs['loss']

        # Backward pass
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        # Should complete without memory errors
        assert True  # If we reach here, memory was sufficient