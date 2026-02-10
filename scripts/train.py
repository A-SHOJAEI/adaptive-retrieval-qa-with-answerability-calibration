#!/usr/bin/env python3
"""Training script for Adaptive Retrieval QA with Answerability Calibration.

This script provides a complete training pipeline with data loading, preprocessing,
model training, validation, and comprehensive logging.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np
import random

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_retrieval_qa_with_answerability_calibration.utils.config import Config
from adaptive_retrieval_qa_with_answerability_calibration.data.loader import DatasetLoader
from adaptive_retrieval_qa_with_answerability_calibration.data.preprocessing import DataPreprocessor
from adaptive_retrieval_qa_with_answerability_calibration.models.model import AdaptiveRetrievalQAModel
from adaptive_retrieval_qa_with_answerability_calibration.training.trainer import AdaptiveQATrainer


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        log_level: Logging level to use.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Adaptive Retrieval QA Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/trained_model",
        help="Output directory for trained model"
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=-1,
        help="Number of training examples to use (-1 for all)"
    )

    parser.add_argument(
        "--val-size",
        type=int,
        default=-1,
        help="Number of validation examples to use (-1 for all)"
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (overrides config)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda/auto)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a quick test with small datasets"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation during training"
    )

    return parser.parse_args()


def load_and_preprocess_data(
    config: Config,
    train_size: int = -1,
    val_size: int = -1,
    quick_test: bool = False
) -> tuple:
    """Load and preprocess datasets.

    Args:
        config: Configuration object.
        train_size: Number of training examples (-1 for all).
        val_size: Number of validation examples (-1 for all).
        quick_test: Whether to run a quick test with small data.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, preprocessor).
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading datasets...")

    # Override sizes for quick test
    if quick_test:
        train_size = 100
        val_size = 50
        logger.info("Quick test mode: using small datasets")

    # Update config with sizes
    if train_size > 0:
        config.set('data.train_size', train_size)
    if val_size > 0:
        config.set('data.val_size', val_size)

    # Initialize data loader
    data_loader = DatasetLoader(config)

    try:
        # Load SQuAD 2.0
        logger.info("Loading SQuAD 2.0 dataset...")
        squad_data = data_loader.load_squad_v2()

        # Load MS MARCO (smaller subset for training efficiency)
        logger.info("Loading MS MARCO dataset...")
        try:
            marco_data = data_loader.load_ms_marco()
        except Exception as e:
            logger.warning(f"Failed to load MS MARCO: {e}")
            logger.info("Proceeding with SQuAD 2.0 only")
            marco_data = None

        # Helper to add missing columns for SQuAD-only data
        def add_squad_columns(dataset):
            """Add is_answerable, source, relevance_score, passage_id to SQuAD data."""
            def add_cols(example, idx):
                example['is_answerable'] = len(example['answers']['text']) > 0
                example['source'] = 'squad_v2'
                example['relevance_score'] = 1.0
                example['passage_id'] = f"squad_{idx}"
                return example
            return dataset.map(add_cols, with_indices=True, desc="Adding columns")

        # Split datasets
        if marco_data:
            # Combine datasets
            train_dataset = data_loader.create_combined_dataset(
                squad_data['train'],
                marco_data['train'] if 'train' in marco_data else marco_data
            )
            val_dataset = add_squad_columns(squad_data['validation'])
        else:
            # Use SQuAD only - add required columns
            train_dataset = add_squad_columns(squad_data['train'])
            val_dataset = add_squad_columns(squad_data['validation'])

        test_dataset = val_dataset  # Use validation as test for this example

        # Log dataset statistics
        if hasattr(data_loader, 'get_dataset_statistics'):
            train_stats = data_loader.get_dataset_statistics(train_dataset)
            logger.info(f"Training dataset statistics: {train_stats}")

    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        # Fallback: create dummy datasets for testing
        logger.warning("Creating dummy datasets for testing")
        from datasets import Dataset

        dummy_data = {
            'question': ['What is AI?'] * (train_size if train_size > 0 else 100),
            'context': ['AI is artificial intelligence.'] * (train_size if train_size > 0 else 100),
            'answers': [{'text': ['AI'], 'answer_start': [0]}] * (train_size if train_size > 0 else 100),
            'is_answerable': [True] * (train_size if train_size > 0 else 100),
            'source': ['dummy'] * (train_size if train_size > 0 else 100),
            'passage_id': [f'dummy_{i}' for i in range(train_size if train_size > 0 else 100)],
            'relevance_score': [1.0] * (train_size if train_size > 0 else 100)
        }

        train_dataset = Dataset.from_dict(dummy_data)
        val_dataset = Dataset.from_dict({k: v[:50] for k, v in dummy_data.items()})
        test_dataset = val_dataset

    # Initialize preprocessor
    logger.info("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(config)

    # Preprocess datasets
    logger.info("Preprocessing training dataset...")
    train_dataset = preprocessor.preprocess_dataset(train_dataset, is_training=True)

    logger.info("Preprocessing validation dataset...")
    val_dataset = preprocessor.preprocess_dataset(val_dataset, is_training=False)

    logger.info("Preprocessing test dataset...")
    test_dataset = preprocessor.preprocess_dataset(test_dataset, is_training=False)

    logger.info("Data loading and preprocessing completed")

    return train_dataset, val_dataset, test_dataset, preprocessor


def initialize_model(config: Config) -> AdaptiveRetrievalQAModel:
    """Initialize the model.

    Args:
        config: Configuration object.

    Returns:
        Initialized model.
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing model...")

    try:
        model = AdaptiveRetrievalQAModel(config)
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Model initialized with {num_params:,} total parameters")
        logger.info(f"Trainable parameters: {num_trainable:,}")

        return model

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise


def main() -> None:
    """Main training function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Adaptive Retrieval QA Training")
    logger.info(f"Arguments: {args}")

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = Config(args.config)

        # Override config with command line arguments
        if args.epochs is not None:
            config.set('training.num_epochs', args.epochs)
        if args.batch_size is not None:
            config.set('training.batch_size', args.batch_size)
        if args.learning_rate is not None:
            config.set('training.learning_rate', args.learning_rate)
        if args.device is not None:
            config.set('infrastructure.device', args.device)
        if args.seed is not None:
            config.set('seed', args.seed)

        # Set random seeds
        seed = config.get('seed', 42)
        set_random_seeds(seed)
        logger.info(f"Random seed set to {seed}")

        # Load and preprocess data
        train_dataset, val_dataset, test_dataset, preprocessor = load_and_preprocess_data(
            config, args.train_size, args.val_size, args.quick_test
        )

        # Skip validation dataset if requested
        if args.skip_validation:
            val_dataset = None
            logger.info("Validation skipped as requested")

        # Initialize model
        model = initialize_model(config)

        # Build passage index for retrieval
        if hasattr(model, 'build_passage_index'):
            logger.info("Building passage retrieval index...")
            try:
                passages, embeddings = preprocessor.create_retrieval_corpus(train_dataset)
                model.build_passage_index(passages, embeddings)
                logger.info(f"Built retrieval index with {len(passages)} passages")
            except Exception as e:
                logger.warning(f"Failed to build retrieval index: {e}")

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = AdaptiveQATrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )

        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming training from {args.resume_from}")
            # Note: Checkpoint resuming would be implemented here
            logger.warning("Checkpoint resuming not implemented in this example")

        # Start training
        logger.info("Starting training...")
        training_history = trainer.train()

        # Save final model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving trained model to {output_dir}")
        model.save_pretrained(str(output_dir))

        # Save configuration
        config.save(output_dir / "config.yaml")

        # Save training history
        import json
        with open(output_dir / "training_history.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [float(x) if hasattr(x, 'item') else x for x in value]
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)

        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {output_dir}")

        # Print summary
        if training_history['train_loss']:
            final_train_loss = training_history['train_loss'][-1]
            final_val_loss = training_history['val_loss'][-1] if training_history['val_loss'] else None

            logger.info("=== Training Summary ===")
            logger.info(f"Final training loss: {final_train_loss:.4f}")
            if final_val_loss is not None:
                logger.info(f"Final validation loss: {final_val_loss:.4f}")

            if training_history['answerability_accuracy']:
                final_accuracy = training_history['answerability_accuracy'][-1]
                logger.info(f"Final answerability accuracy: {final_accuracy:.4f}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()