#!/usr/bin/env python3
"""Prediction script for Adaptive Retrieval QA with Answerability Calibration.

This script loads a trained model and performs inference on input questions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_retrieval_qa_with_answerability_calibration.utils.config import Config
from adaptive_retrieval_qa_with_answerability_calibration.models.model import AdaptiveRetrievalQAModel


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
        description="Run inference with Adaptive Retrieval QA Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/trained_model",
        help="Path to trained model directory"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (if different from model config)"
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to answer"
    )

    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="JSON file containing list of questions"
    )

    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context passage (otherwise retrieved automatically)"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for predictions (JSON format)"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Confidence threshold for answerability (overrides config)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu/cuda/auto)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed prediction information"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    return parser.parse_args()


def load_model_and_config(
    model_path: str,
    config_path: Optional[str] = None,
    confidence_threshold: Optional[float] = None
) -> tuple:
    """Load trained model and configuration.

    Args:
        model_path: Path to the trained model directory.
        config_path: Optional path to configuration file.
        confidence_threshold: Optional confidence threshold override.

    Returns:
        Tuple of (model, config).
    """
    logger = logging.getLogger(__name__)
    model_dir = Path(model_path)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

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

    # Override confidence threshold if provided
    if confidence_threshold is not None:
        config.set('model.confidence_threshold', confidence_threshold)

    logger.info(f"Loading model from {model_path}")

    try:
        # Load the trained model
        model = AdaptiveRetrievalQAModel.from_pretrained(str(model_dir), config)
        logger.info("Model loaded successfully")

        return model, config

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_questions(
    question: Optional[str] = None,
    questions_file: Optional[str] = None
) -> List[str]:
    """Load questions from command line or file.

    Args:
        question: Single question from command line.
        questions_file: Path to JSON file with questions.

    Returns:
        List of questions.
    """
    if question:
        return [question]

    if questions_file:
        with open(questions_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'questions' in data:
                return data['questions']
            else:
                raise ValueError("Invalid questions file format. Expected list or dict with 'questions' key.")

    return []


def predict_single(
    model: AdaptiveRetrievalQAModel,
    question: str,
    context: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run prediction on a single question.

    Args:
        model: Trained model.
        question: Question text.
        context: Optional context passage.
        verbose: Whether to include detailed information.

    Returns:
        Prediction results dictionary.
    """
    logger = logging.getLogger(__name__)

    try:
        # Run prediction
        if context:
            result = model.answer_with_context(question, context)
        else:
            result = model.predict(question)

        # Format output
        output = {
            'question': question,
            'answer': result.get('answer', 'No answer found'),
            'confidence': float(result.get('confidence', 0.0)),
            'is_answerable': bool(result.get('is_answerable', False))
        }

        if verbose:
            output['retrieved_passages'] = result.get('retrieved_passages', [])
            output['retrieval_scores'] = [float(s) for s in result.get('retrieval_scores', [])]
            output['answer_start'] = result.get('answer_start')
            output['answer_end'] = result.get('answer_end')

        return output

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            'question': question,
            'answer': 'Error',
            'confidence': 0.0,
            'is_answerable': False,
            'error': str(e)
        }


def print_prediction(prediction: Dict[str, Any], verbose: bool = False) -> None:
    """Print prediction results in a readable format.

    Args:
        prediction: Prediction results.
        verbose: Whether to print detailed information.
    """
    print("\n" + "=" * 80)
    print(f"Question: {prediction['question']}")
    print("-" * 80)

    if 'error' in prediction:
        print(f"ERROR: {prediction['error']}")
    else:
        print(f"Answer: {prediction['answer']}")
        print(f"Confidence: {prediction['confidence']:.3f}")
        print(f"Answerable: {'Yes' if prediction['is_answerable'] else 'No'}")

        if verbose and 'retrieved_passages' in prediction:
            print("\nRetrieved Passages:")
            for i, passage in enumerate(prediction['retrieved_passages'][:3], 1):
                score = prediction['retrieval_scores'][i-1] if i-1 < len(prediction['retrieval_scores']) else 0.0
                print(f"  {i}. [Score: {score:.3f}] {passage[:200]}...")

    print("=" * 80)


def interactive_mode(model: AdaptiveRetrievalQAModel, verbose: bool = False) -> None:
    """Run model in interactive mode.

    Args:
        model: Trained model.
        verbose: Whether to print detailed information.
    """
    print("\n" + "=" * 80)
    print("Adaptive Retrieval QA - Interactive Mode")
    print("=" * 80)
    print("Enter questions to get answers. Type 'quit' or 'exit' to stop.")
    print("Type 'help' for available commands.")
    print("=" * 80 + "\n")

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break

            if question.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Enter a question to get an answer")
                print("  - 'verbose on/off' - Toggle verbose output")
                print("  - 'help' - Show this help message")
                print("  - 'quit' or 'exit' - Exit interactive mode")
                continue

            if question.lower().startswith('verbose'):
                parts = question.split()
                if len(parts) == 2:
                    if parts[1].lower() == 'on':
                        verbose = True
                        print("Verbose mode enabled")
                    elif parts[1].lower() == 'off':
                        verbose = False
                        print("Verbose mode disabled")
                continue

            # Run prediction
            prediction = predict_single(model, question, verbose=verbose)
            print_prediction(prediction, verbose)

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main() -> None:
    """Main prediction function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Adaptive Retrieval QA Prediction")

    try:
        # Load model and configuration
        model, config = load_model_and_config(
            args.model_path,
            args.config,
            args.confidence_threshold
        )

        # Setup device
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        model.to(device)
        model.eval()

        logger.info(f"Using device: {device}")

        # Interactive mode
        if args.interactive:
            interactive_mode(model, args.verbose)
            return

        # Load questions
        questions = load_questions(args.question, args.questions_file)

        if not questions:
            # Demo mode with example questions
            logger.info("No questions provided. Running demo with example questions.")
            questions = [
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?",
                "What is the speed of light?",
                "When was the Declaration of Independence signed?",
                "What is photosynthesis?"
            ]

        # Run predictions
        predictions = []
        for question in questions:
            logger.info(f"Processing: {question}")
            prediction = predict_single(model, question, args.context, args.verbose)
            predictions.append(prediction)
            print_prediction(prediction, args.verbose)

        # Save results if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"Predictions saved to {args.output_file}")

        logger.info("Prediction completed successfully!")

    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
