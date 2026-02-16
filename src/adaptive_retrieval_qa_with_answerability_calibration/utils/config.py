"""Configuration management utilities."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
import torch


class Config:
    """Configuration manager for loading and accessing configuration values.

    This class provides a convenient interface for loading configuration from
    YAML files and accessing nested configuration values using dot notation.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to the configuration YAML file. If None, loads
                from the default location.
        """
        self.logger = logging.getLogger(__name__)

        if config_path is None:
            # Default to configs/default.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "configs" / "default.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_logging()
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing configuration values.

        Raises:
            FileNotFoundError: If configuration file does not exist.
            yaml.YAMLError: If configuration file is not valid YAML.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Set device automatically if configured as 'auto'
            if config.get('infrastructure', {}).get('device') == 'auto':
                config['infrastructure']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

            return config

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

        # Validate configuration after loading
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate critical configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate confidence threshold
        threshold = self.get('model.confidence_threshold', 0.5)
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError(f"model.confidence_threshold must be in [0,1], got {threshold}")

        # Validate learning rate
        lr = self.get('training.learning_rate', 2e-5)
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"training.learning_rate must be positive, got {lr}")

        # Validate batch size
        batch_size = self.get('training.batch_size', 16)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"training.batch_size must be positive integer, got {batch_size}")

        # Validate top_k retrieval
        top_k = self.get('model.retrieval_top_k', 10)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"model.retrieval_top_k must be positive integer, got {top_k}")

        # Validate max sequence length
        max_len = self.get('model.max_seq_length', 512)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 2048:
            raise ValueError(f"model.max_seq_length must be in (0, 2048], got {max_len}")

        self.logger.debug("Configuration validation passed")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.get('infrastructure.log_level', 'INFO').upper()

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _create_directories(self) -> None:
        """Create necessary directories specified in configuration."""
        directories = [
            self.get('paths.data_dir', './data'),
            self.get('paths.model_dir', './models'),
            self.get('paths.checkpoint_dir', './checkpoints'),
            self.get('paths.log_dir', './logs'),
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'model.learning_rate').
            default: Default value to return if key is not found.

        Returns:
            Configuration value or default.

        Example:
            >>> config = Config()
            >>> batch_size = config.get('training.batch_size', 32)
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Args:
            key: Configuration key in dot notation.
            value: Value to set.
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to YAML file.

        Args:
            output_path: Path to save configuration. If None, overwrites
                the original configuration file.
        """
        if output_path is None:
            output_path = self.config_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to {output_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator."""
        return self.get(key) is not None