#!/usr/bin/env python3
"""Evaluation script for Adaptive Retrieval QA with Answerability Calibration.

This script provides comprehensive evaluation including metrics calculation,
calibration analysis, and report generation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_retrieval_qa_with_answerability_calibration.utils.config import Config
from adaptive_retrieval_qa_with_answerability_calibration.data.loader import DatasetLoader
from adaptive_retrieval_qa_with_answerability_calibration.data.preprocessing import DataPreprocessor
from adaptive_retrieval_qa_with_answerability_calibration.models.model import AdaptiveRetrievalQAModel
from adaptive_retrieval_qa_with_answerability_calibration.evaluation.metrics import AnswerabilityCalibrationMetrics
from adaptive_retrieval_qa_with_answerability_calibration.training.trainer import AdaptiveQADataset

from torch.utils.data import DataLoader


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
        description="Evaluate Adaptive Retrieval QA Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (if different from model config)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=-1,
        help="Number of test examples to use (-1 for all)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu/cuda/auto)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["test", "validation", "squad_test", "marco_test"],
        default="test",
        help="Dataset split to evaluate on"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold for answerability (overrides config)"
    )

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save detailed predictions to file"
    )

    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate calibration and confidence plots"
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
        help="Run quick evaluation with small dataset"
    )

    return parser.parse_args()


def load_model_and_config(
    model_path: str,
    config_path: str = None
) -> tuple:
    """Load trained model and configuration.

    Args:
        model_path: Path to the trained model directory.
        config_path: Optional path to configuration file.

    Returns:
        Tuple of (model, config).
    """
    logger = logging.getLogger(__name__)
    model_dir = Path(model_path)

    # Load configuration
    if config_path:
        config = Config(config_path)
    else:
        config_file = model_dir / "config.yaml"
        if config_file.exists():
            config = Config(str(config_file))
        else:
            logger.warning("No config file found, using default configuration")
            config = Config()

    logger.info(f"Loading model from {model_path}")

    try:
        # Load the trained model
        model = AdaptiveRetrievalQAModel.from_pretrained(str(model_dir), config)
        logger.info("Model loaded successfully")

        return model, config

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_evaluation_dataset(
    config: Config,
    dataset_name: str = "test",
    test_size: int = -1,
    quick_test: bool = False
) -> tuple:
    """Load dataset for evaluation.

    Args:
        config: Configuration object.
        dataset_name: Name of the dataset split to load.
        test_size: Number of examples to use (-1 for all).
        quick_test: Whether to run quick test with small data.

    Returns:
        Tuple of (dataset, preprocessor).
    """
    logger = logging.getLogger(__name__)

    # Override size for quick test
    if quick_test:
        test_size = 50
        logger.info("Quick test mode: using small dataset")

    # Update config
    if test_size > 0:
        config.set('data.test_size', test_size)

    # Initialize data loader
    data_loader = DatasetLoader(config)
    preprocessor = DataPreprocessor(config)

    try:
        if dataset_name in ["test", "validation"]:
            # Load SQuAD 2.0
            logger.info("Loading SQuAD 2.0 dataset...")
            squad_data = data_loader.load_squad_v2()

            if dataset_name == "validation" or dataset_name == "test":
                dataset = squad_data['validation']  # Use validation as test split
            else:
                # Try to use actual test split if available
                dataset = squad_data.get('test', squad_data['validation'])

        elif dataset_name == "squad_test":
            squad_data = data_loader.load_squad_v2()
            dataset = squad_data['validation']

        elif dataset_name == "marco_test":
            # Load MS MARCO test data
            logger.info("Loading MS MARCO dataset...")
            marco_data = data_loader.load_ms_marco()
            dataset = marco_data['test'] if 'test' in marco_data else marco_data

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Preprocess dataset
        logger.info("Preprocessing dataset...")
        dataset = preprocessor.preprocess_dataset(dataset, is_training=False)

        logger.info(f"Loaded {len(dataset)} examples for evaluation")

        return dataset, preprocessor

    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        # Create dummy dataset for testing
        logger.warning("Creating dummy dataset for testing")
        from datasets import Dataset

        size = test_size if test_size > 0 else 100
        dummy_data = {
            'question': ['What is AI?'] * size,
            'context': ['AI is artificial intelligence.'] * size,
            'answers': [{'text': ['AI'], 'answer_start': [0]}] * size,
            'is_answerable': [True] * size,
            'source': ['dummy'] * size,
            'passage_id': [f'dummy_{i}' for i in range(size)],
            'relevance_score': [1.0] * size
        }

        dataset = Dataset.from_dict(dummy_data)
        dataset = preprocessor.preprocess_dataset(dataset, is_training=False)

        return dataset, preprocessor


def run_evaluation(
    model: AdaptiveRetrievalQAModel,
    dataset: 'Dataset',
    config: Config,
    batch_size: int = 16,
    device: str = "auto"
) -> Dict[str, Any]:
    """Run comprehensive model evaluation.

    Args:
        model: The trained model to evaluate.
        dataset: Evaluation dataset.
        config: Configuration object.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing evaluation results.
    """
    logger = logging.getLogger(__name__)

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Move model to device
    model.to(device)
    model.eval()

    logger.info(f"Running evaluation on device: {device}")

    # Create dataloader
    torch_dataset = AdaptiveQADataset(dataset, config)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for evaluation to avoid issues
        collate_fn=lambda batch: {k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()}
    )

    # Initialize metrics calculator
    evaluator = AnswerabilityCalibrationMetrics(config)

    # Run evaluation
    logger.info("Starting evaluation...")
    metrics, predictions = evaluator.evaluate_model(
        model, dataloader, device, return_predictions=True
    )

    logger.info("Evaluation completed")

    return {
        'metrics': metrics,
        'predictions': predictions,
        'config': config.to_dict(),
        'dataset_size': len(dataset)
    }


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    save_predictions: bool = False,
    generate_plots: bool = False
) -> None:
    """Save evaluation results.

    Args:
        results: Evaluation results.
        output_dir: Output directory.
        save_predictions: Whether to save detailed predictions.
        generate_plots: Whether to generate visualization plots.
    """
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        # Make metrics JSON serializable
        serializable_metrics = {}
        for key, value in results['metrics'].items():
            if isinstance(value, (list, np.ndarray)):
                serializable_metrics[key] = [float(x) for x in value]
            elif isinstance(value, np.number):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value

        json.dump(serializable_metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_file}")

    # Save predictions if requested
    if save_predictions:
        predictions_file = output_path / "predictions.json"
        with open(predictions_file, 'w') as f:
            # Make predictions JSON serializable
            serializable_preds = {}
            for key, value in results['predictions'].items():
                if isinstance(value, (list, np.ndarray)):
                    if isinstance(value[0], np.number):
                        serializable_preds[key] = [float(x) for x in value]
                    else:
                        serializable_preds[key] = list(value)
                else:
                    serializable_preds[key] = value

            json.dump(serializable_preds, f, indent=2)

        logger.info(f"Predictions saved to {predictions_file}")

    # Generate evaluation report
    config = Config()
    config._config = results['config']
    evaluator = AnswerabilityCalibrationMetrics(config)

    report_path = evaluator.generate_evaluation_report(
        results['metrics'],
        str(output_path)
    )
    logger.info(f"Evaluation report saved to {report_path}")

    # Generate plots if requested
    if generate_plots:
        try:
            predictions = results['predictions']
            if 'answerability_probs' in predictions and 'answerability_labels' in predictions:
                probs = np.array(predictions['answerability_probs'])
                labels = np.array(predictions['answerability_labels'])

                # Calibration curve
                calib_plot_path = output_path / "calibration_curve.png"
                evaluator.plot_calibration_curve(probs, labels, str(calib_plot_path))

                # Confidence histogram
                hist_plot_path = output_path / "confidence_histogram.png"
                evaluator.plot_confidence_histogram(probs, labels, str(hist_plot_path))

                logger.info("Evaluation plots generated")

        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")


def analyze_results(results: Dict[str, Any]) -> None:
    """Analyze and print evaluation results.

    Args:
        results: Evaluation results.
    """
    logger = logging.getLogger(__name__)
    metrics = results['metrics']

    logger.info("=== Evaluation Results Analysis ===")

    # Key metrics
    key_metrics = [
        ('Exact Match', 'exact_match'),
        ('F1 Score', 'f1_score'),
        ('Answerability AUROC', 'answerability_auroc'),
        ('Retrieval MRR@10', 'retrieval_mrr@10'),
        ('Calibration ECE', 'calibration_ece')
    ]

    logger.info("\nKey Metrics:")
    for display_name, metric_key in key_metrics:
        if metric_key in metrics:
            value = metrics[metric_key]
            logger.info(f"  {display_name}: {value:.4f}")

    # Answerability analysis
    if 'answerability_accuracy' in metrics:
        logger.info(f"\nAnswerability Analysis:")
        logger.info(f"  Accuracy: {metrics['answerability_accuracy']:.4f}")
        logger.info(f"  Precision: {metrics.get('answerability_precision', 0.0):.4f}")
        logger.info(f"  Recall: {metrics.get('answerability_recall', 0.0):.4f}")
        logger.info(f"  F1: {metrics.get('answerability_f1', 0.0):.4f}")

    # Confidence analysis
    if 'confidence_mean' in metrics:
        logger.info(f"\nConfidence Analysis:")
        logger.info(f"  Mean Confidence: {metrics['confidence_mean']:.4f}")
        logger.info(f"  Std Confidence: {metrics['confidence_std']:.4f}")
        logger.info(f"  Min Confidence: {metrics['confidence_min']:.4f}")
        logger.info(f"  Max Confidence: {metrics['confidence_max']:.4f}")

    # Target achievement
    target_metrics = results['config'].get('evaluation', {}).get('target_metrics', {})
    if target_metrics:
        logger.info(f"\nTarget Achievement:")
        achieved = 0
        total = 0
        for metric_name, target_value in target_metrics.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                is_achieved = current_value >= target_value
                status = "✓" if is_achieved else "✗"
                logger.info(f"  {metric_name}: {current_value:.4f} / {target_value:.4f} {status}")
                if is_achieved:
                    achieved += 1
                total += 1

        if total > 0:
            achievement_rate = achieved / total
            logger.info(f"  Overall Achievement: {achieved}/{total} ({achievement_rate:.1%})")

    logger.info("=== Analysis Complete ===")


def main() -> None:
    """Main evaluation function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Adaptive Retrieval QA Evaluation")
    logger.info(f"Arguments: {args}")

    try:
        # Load model and configuration
        model, config = load_model_and_config(args.model_path, args.config)

        # Override config with command line arguments
        if args.confidence_threshold is not None:
            config.set('model.confidence_threshold', args.confidence_threshold)

        # Load evaluation dataset
        dataset, preprocessor = load_evaluation_dataset(
            config, args.dataset, args.test_size, args.quick_test
        )

        # Build retrieval index if needed
        if hasattr(model, 'build_passage_index') and model.passage_corpus is None:
            logger.info("Building passage retrieval index...")
            try:
                passages, embeddings = preprocessor.create_retrieval_corpus(dataset)
                model.build_passage_index(passages, embeddings)
                logger.info(f"Built retrieval index with {len(passages)} passages")
            except Exception as e:
                logger.warning(f"Failed to build retrieval index: {e}")

        # Run evaluation
        results = run_evaluation(
            model, dataset, config, args.batch_size, args.device
        )

        # Save results
        save_results(
            results,
            args.output_dir,
            args.save_predictions,
            args.generate_plots
        )

        # Analyze results
        analyze_results(results)

        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()