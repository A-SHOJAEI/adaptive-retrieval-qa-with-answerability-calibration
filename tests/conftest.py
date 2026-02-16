"""PyTest configuration and fixtures for the test suite."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import torch
import numpy as np
from datasets import Dataset

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_retrieval_qa_with_answerability_calibration.utils.config import Config
from adaptive_retrieval_qa_with_answerability_calibration.data.loader import DatasetLoader
from adaptive_retrieval_qa_with_answerability_calibration.data.preprocessing import DataPreprocessor
from adaptive_retrieval_qa_with_answerability_calibration.models.model import AdaptiveRetrievalQAModel


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests.

    Yields:
        Path to temporary directory.
    """
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def test_config(temp_dir: Path) -> Config:
    """Create test configuration.

    Args:
        temp_dir: Temporary directory for test files.

    Returns:
        Test configuration object.
    """
    config_data = {
        'model': {
            'retriever_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'reader_name': 'distilbert-base-uncased-distilled-squad',
            'confidence_threshold': 0.5,
            'max_seq_length': 256,  # Shorter for faster testing
            'retrieval_top_k': 5,  # Fewer for faster testing
            'answer_max_length': 50
        },
        'training': {
            'batch_size': 4,  # Small batch for testing
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'warmup_steps': 10,
            'weight_decay': 0.01,
            'gradient_clip_norm': 1.0,
            'save_steps': 100,
            'eval_steps': 50,
            'logging_steps': 10,
            'early_stopping_patience': 2
        },
        'calibration': {
            'temperature_scaling': True,
            'platt_scaling': False,
            'isotonic_regression': False,
            'confidence_bins': 5  # Fewer bins for testing
        },
        'data': {
            'train_size': 20,  # Small datasets for testing
            'val_size': 10,
            'test_size': 10,
            'max_passages_per_question': 3,
            'passage_max_length': 100
        },
        'evaluation': {
            'target_metrics': {
                'exact_match': 0.5,
                'f1_score': 0.6,
                'answerability_auroc': 0.8,
                'retrieval_mrr_at_10': 0.3,
                'calibration_ece': 0.1
            }
        },
        'infrastructure': {
            'device': 'cpu',  # Force CPU for testing
            'mixed_precision': False,
            'num_workers': 0,
            'pin_memory': False
        },
        'mlflow': {
            'experiment_name': 'test-adaptive-retrieval-qa',
            'tracking_uri': str(temp_dir / 'mlruns'),
            'log_artifacts': False
        },
        'seed': 42,
        'deterministic': True,
        'paths': {
            'data_dir': str(temp_dir / 'data'),
            'model_dir': str(temp_dir / 'models'),
            'checkpoint_dir': str(temp_dir / 'checkpoints'),
            'log_dir': str(temp_dir / 'logs')
        }
    }

    # Create temporary config file
    config_path = temp_dir / "test_config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)

    return Config(str(config_path))


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a sample dataset for testing.

    Returns:
        Sample dataset with test data.
    """
    # Create diverse test examples
    data = {
        'question': [
            'What is the capital of France?',
            'Who invented the telephone?',
            'What is the speed of light?',
            'When did World War II end?',
            'What color is the sky?',
            'How many legs does a spider have?',
            'What is photosynthesis?',
            'Who wrote Romeo and Juliet?',
            'What is the largest planet?',
            'When was Python created?'
        ],
        'context': [
            'Paris is the capital and largest city of France.',
            'Alexander Graham Bell is credited with inventing the telephone in 1876.',
            'The speed of light in vacuum is approximately 299,792,458 meters per second.',
            'World War II ended in 1945 when Japan surrendered.',
            'The sky appears blue due to Rayleigh scattering of light.',
            'Spiders are arachnids with eight legs and two body segments.',
            'Photosynthesis is the process by which plants convert sunlight into energy.',
            'Romeo and Juliet was written by William Shakespeare.',
            'Jupiter is the largest planet in our solar system.',
            'Python was created by Guido van Rossum and first released in 1991.'
        ],
        'answers': [
            {'text': ['Paris'], 'answer_start': [0]},
            {'text': ['Alexander Graham Bell'], 'answer_start': [0]},
            {'text': ['299,792,458 meters per second'], 'answer_start': [49]},
            {'text': ['1945'], 'answer_start': [23]},
            {'text': ['blue'], 'answer_start': [17]},
            {'text': ['eight'], 'answer_start': [24]},
            {'text': ['process by which plants convert sunlight into energy'], 'answer_start': [19]},
            {'text': ['William Shakespeare'], 'answer_start': [36]},
            {'text': ['Jupiter'], 'answer_start': [0]},
            {'text': ['1991'], 'answer_start': [68]}
        ],
        'is_answerable': [True] * 8 + [False, False],  # Last 2 are unanswerable for testing
        'source': ['test'] * 10,
        'passage_id': [f'test_{i}' for i in range(10)],
        'relevance_score': [0.9, 0.8, 0.95, 0.85, 0.7, 0.9, 0.85, 0.8, 0.6, 0.5]
    }

    return Dataset.from_dict(data)


@pytest.fixture
def small_dataset() -> Dataset:
    """Create a very small dataset for quick testing.

    Returns:
        Small dataset with minimal test data.
    """
    data = {
        'question': ['What is AI?', 'Who created Python?'],
        'context': ['AI stands for artificial intelligence.', 'Python was created by Guido van Rossum.'],
        'answers': [
            {'text': ['artificial intelligence'], 'answer_start': [14]},
            {'text': ['Guido van Rossum'], 'answer_start': [22]}
        ],
        'is_answerable': [True, True],
        'source': ['test', 'test'],
        'passage_id': ['test_0', 'test_1'],
        'relevance_score': [1.0, 0.9]
    }

    return Dataset.from_dict(data)


@pytest.fixture
def data_loader(test_config: Config) -> DatasetLoader:
    """Create data loader for testing.

    Args:
        test_config: Test configuration.

    Returns:
        DatasetLoader instance.
    """
    return DatasetLoader(test_config)


@pytest.fixture
def preprocessor(test_config: Config) -> DataPreprocessor:
    """Create data preprocessor for testing.

    Args:
        test_config: Test configuration.

    Returns:
        DataPreprocessor instance.
    """
    return DataPreprocessor(test_config)


@pytest.fixture
def model(test_config: Config) -> AdaptiveRetrievalQAModel:
    """Create model for testing.

    Args:
        test_config: Test configuration.

    Returns:
        AdaptiveRetrievalQAModel instance.
    """
    return AdaptiveRetrievalQAModel(test_config)


@pytest.fixture
def processed_dataset(sample_dataset: Dataset, preprocessor: DataPreprocessor) -> Dataset:
    """Create processed dataset for testing.

    Args:
        sample_dataset: Raw sample dataset.
        preprocessor: Data preprocessor.

    Returns:
        Processed dataset.
    """
    return preprocessor.preprocess_dataset(sample_dataset, is_training=True)


@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducible testing."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture
def dummy_batch() -> Dict[str, torch.Tensor]:
    """Create dummy batch for testing.

    Returns:
        Dictionary containing dummy batch tensors.
    """
    batch_size = 2
    seq_len = 64

    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'token_type_ids': torch.zeros(batch_size, seq_len),
        'start_positions': torch.tensor([5, 10]),
        'end_positions': torch.tensor([8, 15]),
        'is_answerable': torch.tensor([1.0, 0.0]),
        'retrieval_scores': torch.tensor([0.8, 0.6]),
        'confidence_features': torch.randn(batch_size, 10)
    }


class MockMLflowRun:
    """Mock MLflow run for testing."""

    def __init__(self):
        self.params = {}
        self.metrics = {}
        self.artifacts = []

    def log_param(self, key, value):
        self.params[key] = value

    def log_metric(self, key, value, step=None):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append((value, step))

    def log_artifact(self, path):
        self.artifacts.append(path)


@pytest.fixture
def mock_mlflow_run():
    """Create mock MLflow run for testing.

    Returns:
        Mock MLflow run object.
    """
    return MockMLflowRun()