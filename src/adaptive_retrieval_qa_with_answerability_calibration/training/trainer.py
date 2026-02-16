"""Advanced training infrastructure with MLflow integration and early stopping.

This module provides a comprehensive training framework for the adaptive
retrieval QA system with extensive logging, checkpointing, and monitoring.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR,
    CosineAnnealingLR
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.model import AdaptiveRetrievalQAModel
from ..evaluation.metrics import AnswerabilityCalibrationMetrics
from ..utils.config import Config


class AdaptiveQADataset(Dataset):
    """PyTorch Dataset for the adaptive QA system."""

    def __init__(self, dataset: 'Dataset', config: Config) -> None:
        """Initialize dataset.

        Args:
            dataset: Hugging Face dataset with preprocessed examples.
            config: Configuration object.
        """
        self.dataset = dataset
        self.config = config

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example.

        Args:
            idx: Index of the example.

        Returns:
            Dictionary containing tensors for model input.
        """
        example = self.dataset[idx]

        # Convert to tensors
        item = {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.long),
            'is_answerable': torch.tensor(example['is_answerable'], dtype=torch.float),
        }

        # Add token type IDs if available
        if 'token_type_ids' in example:
            item['token_type_ids'] = torch.tensor(example['token_type_ids'], dtype=torch.long)

        # Add answer positions for training
        if 'start_positions' in example:
            item['start_positions'] = torch.tensor(example['start_positions'], dtype=torch.long)
            item['end_positions'] = torch.tensor(example['end_positions'], dtype=torch.long)

        # Add confidence features
        if 'confidence_features' in example:
            # Convert dict to tensor
            features = example['confidence_features']
            feature_vector = [
                features.get('question_length', 0.0),
                features.get('context_length', 0.0),
                features.get('question_word_overlap', 0.0),
                features.get('has_question_words', 0.0),
                features.get('context_complexity', 0.0),
                features.get('retrieval_score', 0.0),
                features.get('is_squad', 0.0),
                features.get('is_marco', 0.0),
                0.0,  # Reserved
                0.0   # Reserved
            ]
            item['confidence_features'] = torch.tensor(feature_vector, dtype=torch.float)

        # Add retrieval scores
        if 'relevance_score' in example:
            item['retrieval_scores'] = torch.tensor(example['relevance_score'], dtype=torch.float)

        return item


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(
        self,
        val_loss: float,
        model: nn.Module
    ) -> bool:
        """Check if training should be stopped.

        Args:
            val_loss: Current validation loss.
            model: Model to potentially save weights from.

        Returns:
            True if training should be stopped.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model: nn.Module) -> None:
        """Save current model weights."""
        if self.restore_best_weights:
            import copy
            self.best_weights = copy.deepcopy(model.state_dict())


class AdaptiveQATrainer:
    """Advanced trainer for the Adaptive Retrieval QA system.

    This trainer provides comprehensive training infrastructure with MLflow
    integration, advanced scheduling, early stopping, and detailed monitoring.
    """

    def __init__(
        self,
        model: AdaptiveRetrievalQAModel,
        config: Config,
        train_dataset: Optional['Dataset'] = None,
        val_dataset: Optional['Dataset'] = None,
        test_dataset: Optional['Dataset'] = None
    ) -> None:
        """Initialize trainer.

        Args:
            model: The adaptive QA model to train.
            config: Configuration object.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            test_dataset: Test dataset.
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Training parameters
        self.device = torch.device(config.get('infrastructure.device', 'cpu'))
        self.batch_size = config.get('training.batch_size', 16)
        self.learning_rate = float(config.get('training.learning_rate', 2e-5))
        self.num_epochs = config.get('training.num_epochs', 10)
        self.warmup_steps = config.get('training.warmup_steps', 1000)
        self.weight_decay = config.get('training.weight_decay', 0.01)
        self.gradient_clip_norm = config.get('training.gradient_clip_norm', 1.0)

        # Logging and checkpointing
        self.save_steps = config.get('training.save_steps', 1000)
        self.eval_steps = config.get('training.eval_steps', 500)
        self.logging_steps = config.get('training.logging_steps', 100)

        # Paths
        self.checkpoint_dir = Path(config.get('paths.checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'answerability_accuracy': [],
            'qa_f1': []
        }

        # Early stopping
        patience = config.get('training.early_stopping_patience', 3)
        self.early_stopping = EarlyStopping(patience=patience)

        # Evaluation metrics
        self.evaluator = AnswerabilityCalibrationMetrics(config)

        # Move model to device
        self.model.to(self.device)

        # Initialize optimizer and scheduler
        self._setup_optimizer_and_scheduler()

        # Setup MLflow
        self._setup_mlflow()

    def _setup_optimizer_and_scheduler(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Freeze retriever parameters (pre-trained, used only for encoding)
        for name, param in self.model.named_parameters():
            if 'retriever' in name:
                param.requires_grad = False

        # Separate parameters for different learning rates
        qa_params = []
        calibrator_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'calibrator' in name:
                calibrator_params.append(param)
            else:
                qa_params.append(param)

        # Different learning rates for different components
        qa_lr = float(self.learning_rate)
        calibrator_lr = float(self.learning_rate) * 2  # Higher LR for new calibrator

        optimizer_params = [
            {'params': qa_params, 'lr': qa_lr},
            {'params': calibrator_params, 'lr': calibrator_lr}
        ]

        self.optimizer = AdamW(
            optimizer_params,
            weight_decay=self.weight_decay
        )

        # Calculate total steps for scheduling
        if self.train_dataset is not None:
            total_steps = (len(self.train_dataset) // self.batch_size) * self.num_epochs
        else:
            total_steps = 10000  # Default value

        # Combined scheduler: warmup + cosine annealing
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.1
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )

        self.logger.info(f"Setup optimizer with {total_steps} total steps")

    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment tracking."""
        try:
            experiment_name = self.config.get('mlflow.experiment_name', 'adaptive-retrieval-qa')
            tracking_uri = self.config.get('mlflow.tracking_uri', './mlruns')

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            # Start MLflow run
            mlflow.start_run()

            # Log hyperparameters
            mlflow.log_params({
                'model_name': 'AdaptiveRetrievalQA',
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'warmup_steps': self.warmup_steps,
                'weight_decay': self.weight_decay,
                'qa_model': self.config.get('model.reader_name'),
                'retriever': self.config.get('model.retriever_name'),
                'confidence_threshold': self.config.get('model.confidence_threshold'),
            })

            self.logger.info(f"MLflow tracking initialized: {experiment_name}")

        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {e}")
            self.logger.info("Continuing without MLflow tracking")

    def prepare_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare training and validation dataloaders.

        Returns:
            Tuple of (train_dataloader, val_dataloader).

        Raises:
            ValueError: If training dataset is not provided.
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset is required")

        # Create PyTorch datasets
        train_torch_dataset = AdaptiveQADataset(self.train_dataset, self.config)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_torch_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get('infrastructure.num_workers', 0),
            pin_memory=self.config.get('infrastructure.pin_memory', True),
            collate_fn=self._collate_fn
        )

        val_dataloader = None
        if self.val_dataset is not None:
            val_torch_dataset = AdaptiveQADataset(self.val_dataset, self.config)
            val_dataloader = DataLoader(
                val_torch_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.config.get('infrastructure.num_workers', 0),
                pin_memory=self.config.get('infrastructure.pin_memory', True),
                collate_fn=self._collate_fn
            )

        return train_dataloader, val_dataloader

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching.

        Args:
            batch: List of examples from dataset.

        Returns:
            Batched examples.
        """
        # Stack all tensors
        batched = {}
        for key in batch[0].keys():
            batched[key] = torch.stack([item[key] for item in batch])

        return batched

    def train_epoch(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_dataloader: Training data loader.
            val_dataloader: Optional validation data loader.

        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        total_loss = 0
        total_qa_loss = 0
        total_calibration_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)

            loss = outputs['loss']
            qa_loss = outputs.get('qa_loss', torch.tensor(0.0))
            calibration_loss = outputs.get('calibration_loss', torch.tensor(0.0))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            total_qa_loss += qa_loss.item()
            total_calibration_loss += calibration_loss.item()
            num_batches += 1
            self.current_step += 1

            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })

            # Logging
            if self.current_step % self.logging_steps == 0:
                self._log_training_step(loss.item(), current_lr)

            # Validation
            if val_dataloader and self.current_step % self.eval_steps == 0:
                val_metrics = self.validate(val_dataloader)
                self.model.train()  # Reset to training mode

                # Early stopping check
                if self.early_stopping(val_metrics['loss'], self.model):
                    self.logger.info("Early stopping triggered")
                    break

            # Checkpointing
            if self.current_step % self.save_steps == 0:
                self.save_checkpoint()

        # Epoch metrics
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'qa_loss': total_qa_loss / num_batches,
            'calibration_loss': total_calibration_loss / num_batches,
        }

        return epoch_metrics

    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Run validation.

        Args:
            val_dataloader: Validation data loader.

        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_answerability_preds = []
        all_answerability_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)

                # Accumulate losses
                total_loss += outputs['loss'].item()

                # Collect predictions for metrics
                answerability_probs = outputs['answerability_probs']
                answerability_preds = (answerability_probs > self.config.get('model.confidence_threshold', 0.5))

                all_answerability_preds.extend(answerability_preds.cpu().numpy())
                all_answerability_labels.extend(batch['is_answerable'].cpu().numpy())

                # QA predictions (simplified for validation)
                start_logits = outputs['start_logits']
                end_logits = outputs['end_logits']

                start_preds = torch.argmax(start_logits, dim=-1)
                end_preds = torch.argmax(end_logits, dim=-1)

                if 'start_positions' in batch:
                    start_labels = batch['start_positions']
                    end_labels = batch['end_positions']

                    # Simple exact match for start/end positions
                    exact_matches = (
                        (start_preds == start_labels) & (end_preds == end_labels)
                    ).cpu().numpy()

                    all_predictions.extend(exact_matches)
                    all_labels.extend([1] * len(exact_matches))

        # Calculate metrics
        val_metrics = {
            'loss': total_loss / len(val_dataloader),
        }

        if all_answerability_preds and all_answerability_labels:
            # Answerability accuracy
            answerability_acc = np.mean(
                np.array(all_answerability_preds) == np.array(all_answerability_labels)
            )
            val_metrics['answerability_accuracy'] = answerability_acc

        if all_predictions and all_labels:
            # QA exact match
            qa_exact_match = np.mean(all_predictions)
            val_metrics['qa_exact_match'] = qa_exact_match

        # Log validation metrics
        self._log_validation_metrics(val_metrics)

        return val_metrics

    def train(self) -> Dict[str, List[float]]:
        """Run complete training process.

        Returns:
            Dictionary containing training history.

        Raises:
            ValueError: If training dataset is not set.
        """
        if self.train_dataset is None:
            raise ValueError("Training dataset must be set before training")

        self.logger.info("Starting training")
        self.logger.info(f"Training for {self.num_epochs} epochs")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")

        # Prepare dataloaders
        train_dataloader, val_dataloader = self.prepare_dataloaders()

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, val_dataloader)

            # Validation
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)
                val_loss = val_metrics['loss']
            else:
                val_metrics = {}
                val_loss = train_metrics['loss']

            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rates'].append(self.scheduler.get_last_lr()[0])

            if 'answerability_accuracy' in val_metrics:
                self.training_history['answerability_accuracy'].append(
                    val_metrics['answerability_accuracy']
                )

            if 'qa_exact_match' in val_metrics:
                self.training_history['qa_f1'].append(val_metrics['qa_exact_match'])

            # Log epoch results
            self.logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        self.save_checkpoint(is_final=True)

        # Generate training plots
        self._plot_training_history()

        # Final evaluation
        if self.test_dataset is not None:
            final_metrics = self.evaluate_final()
            # Only log scalar metrics to MLflow (filter out lists/dicts)
            scalar_metrics = {k: v for k, v in final_metrics.items()
                              if isinstance(v, (int, float))}
            try:
                mlflow.log_metrics(scalar_metrics)
            except Exception as e:
                self.logger.warning(f"Failed to log final metrics to MLflow: {e}")

        # End MLflow run
        try:
            mlflow.end_run()
        except Exception as e:
            self.logger.warning(f"Failed to end MLflow run: {e}")

        return self.training_history

    def _log_training_step(self, loss: float, learning_rate: float) -> None:
        """Log training step metrics."""
        try:
            mlflow.log_metrics({
                'train_step_loss': loss,
                'learning_rate': learning_rate
            }, step=self.current_step)
        except Exception as e:
            self.logger.warning(f"Failed to log training metrics: {e}")

    def _log_validation_metrics(self, val_metrics: Dict[str, float]) -> None:
        """Log validation metrics."""
        try:
            mlflow.log_metrics(val_metrics, step=self.current_step)
        except Exception as e:
            self.logger.warning(f"Failed to log validation metrics: {e}")

    def save_checkpoint(self, is_final: bool = False) -> None:
        """Save model checkpoint.

        Args:
            is_final: Whether this is the final checkpoint.
        """
        if is_final:
            checkpoint_path = self.checkpoint_dir / "final_model"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.current_step}"

        # Save model
        self.model.save_pretrained(str(checkpoint_path))

        # Save training state
        state_path = checkpoint_path / "training_state.pt"
        torch.save({
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, state_path)

        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _plot_training_history(self) -> None:
        """Generate and save training history plots."""
        if not self.training_history['train_loss']:
            return

        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        axes[0, 0].plot(epochs, self.training_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Val Loss', color='orange')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Learning rate
        axes[0, 1].plot(epochs, self.training_history['learning_rates'], color='green')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)

        # Answerability accuracy
        if self.training_history['answerability_accuracy']:
            axes[1, 0].plot(epochs, self.training_history['answerability_accuracy'], color='red')
            axes[1, 0].set_title('Answerability Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].grid(True)

        # QA F1
        if self.training_history['qa_f1']:
            axes[1, 1].plot(epochs, self.training_history['qa_f1'], color='purple')
            axes[1, 1].set_title('QA Performance (Exact Match)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Exact Match')
            axes[1, 1].grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = self.checkpoint_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        try:
            mlflow.log_artifact(str(plot_path))
        except Exception as e:
            self.logger.warning(f"Failed to log training plot artifact: {e}")

    def evaluate_final(self) -> Dict[str, float]:
        """Run final comprehensive evaluation on test set.

        Returns:
            Dictionary containing final evaluation metrics.
        """
        if self.test_dataset is None:
            self.logger.warning("No test dataset provided for final evaluation")
            return {}

        self.logger.info("Running final evaluation on test set")

        test_torch_dataset = AdaptiveQADataset(self.test_dataset, self.config)
        test_dataloader = DataLoader(
            test_torch_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

        # Run comprehensive evaluation using the metrics module
        final_metrics = self.evaluator.evaluate_model(
            self.model,
            test_dataloader,
            self.device
        )

        # Log final metrics
        self.logger.info("Final Evaluation Results:")
        for metric, value in final_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")

        return final_metrics