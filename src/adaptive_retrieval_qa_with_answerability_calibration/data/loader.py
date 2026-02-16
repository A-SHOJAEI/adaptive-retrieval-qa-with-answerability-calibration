"""Dataset loading utilities for SQuAD 2.0 and MS MARCO."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np

from ..utils.config import Config


class DatasetLoader:
    """Loads and manages SQuAD 2.0 and MS MARCO datasets.

    This class handles loading of both SQuAD 2.0 (for question answering with
    unanswerable questions) and MS MARCO (for passage retrieval) datasets,
    providing a unified interface for data access.
    """

    def __init__(self, config: Config) -> None:
        """Initialize dataset loader.

        Args:
            config: Configuration object containing dataset parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set random seed for reproducibility
        np.random.seed(config.get('seed', 42))

    def load_squad_v2(
        self,
        split: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        """Load SQuAD 2.0 dataset.

        Args:
            split: Dataset split to load ('train', 'validation', or None for all).

        Returns:
            Dataset or DatasetDict containing SQuAD 2.0 data.
        """
        self.logger.info(f"Loading SQuAD 2.0 dataset (split: {split})")

        try:
            dataset = load_dataset("squad_v2", split=split)

            if split is None:
                # Apply size limits to all splits
                dataset = self._apply_size_limits(dataset, is_squad=True)
            else:
                # Apply size limit to single split
                size_key = f"{split}_size" if split != "validation" else "val_size"
                max_size = self.config.get(f"data.{size_key}", -1)

                if max_size > 0 and len(dataset) > max_size:
                    # Stratified sampling to maintain answerable/unanswerable ratio
                    dataset = self._stratified_sample(dataset, max_size)

            self.logger.info(f"Successfully loaded SQuAD 2.0: {self._get_dataset_info(dataset)}")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading SQuAD 2.0 dataset: {e}")
            raise

    def load_ms_marco(
        self,
        split: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        """Load MS MARCO passage ranking dataset.

        Args:
            split: Dataset split to load ('train', 'validation', or None for all).

        Returns:
            Dataset or DatasetDict containing MS MARCO data.
        """
        self.logger.info(f"Loading MS MARCO dataset (split: {split})")

        try:
            # Load MS MARCO passage ranking dataset
            dataset = load_dataset(
                "ms_marco",
                "v1.1",
                split=split
            )

            if split is None:
                dataset = self._apply_size_limits(dataset, is_squad=False)
            else:
                size_key = f"{split}_size" if split != "validation" else "val_size"
                max_size = self.config.get(f"data.{size_key}", -1)

                if max_size > 0 and len(dataset) > max_size:
                    indices = np.random.choice(
                        len(dataset), max_size, replace=False
                    )
                    dataset = dataset.select(indices)

            self.logger.info(f"Successfully loaded MS MARCO: {self._get_dataset_info(dataset)}")
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading MS MARCO dataset: {e}")
            raise

    def create_combined_dataset(
        self,
        squad_data: Dataset,
        marco_data: Dataset
    ) -> Dataset:
        """Create combined dataset from SQuAD and MS MARCO data.

        Args:
            squad_data: SQuAD 2.0 dataset.
            marco_data: MS MARCO dataset.

        Returns:
            Combined dataset with unified schema.
        """
        self.logger.info("Creating combined dataset from SQuAD 2.0 and MS MARCO")

        try:
            # Process SQuAD data
            squad_processed = self._process_squad_for_combined(squad_data)

            # Process MS MARCO data
            marco_processed = self._process_marco_for_combined(marco_data)

            # Combine datasets
            combined_data = {
                'question': squad_processed['question'] + marco_processed['question'],
                'context': squad_processed['context'] + marco_processed['context'],
                'answers': squad_processed['answers'] + marco_processed['answers'],
                'is_answerable': squad_processed['is_answerable'] + marco_processed['is_answerable'],
                'source': squad_processed['source'] + marco_processed['source'],
                'passage_id': squad_processed['passage_id'] + marco_processed['passage_id'],
                'relevance_score': squad_processed['relevance_score'] + marco_processed['relevance_score']
            }

            combined_dataset = Dataset.from_dict(combined_data)

            # Shuffle the combined dataset
            combined_dataset = combined_dataset.shuffle(seed=self.config.get('seed', 42))

            self.logger.info(f"Created combined dataset with {len(combined_dataset)} examples")
            return combined_dataset

        except Exception as e:
            self.logger.error(f"Error creating combined dataset: {e}")
            raise

    def _apply_size_limits(
        self,
        dataset: DatasetDict,
        is_squad: bool
    ) -> DatasetDict:
        """Apply size limits to all splits in a DatasetDict."""
        limited_dataset = {}

        for split_name, split_data in dataset.items():
            size_key = f"{split_name}_size" if split_name != "validation" else "val_size"
            max_size = self.config.get(f"data.{size_key}", -1)

            if max_size > 0 and len(split_data) > max_size:
                if is_squad and split_name == 'train':
                    # For SQuAD training data, maintain answerable/unanswerable ratio
                    limited_dataset[split_name] = self._stratified_sample(split_data, max_size)
                else:
                    # Random sampling for other splits
                    indices = np.random.choice(
                        len(split_data), max_size, replace=False
                    )
                    limited_dataset[split_name] = split_data.select(indices)
            else:
                limited_dataset[split_name] = split_data

        return DatasetDict(limited_dataset)

    def _stratified_sample(self, dataset: Dataset, max_size: int) -> Dataset:
        """Perform stratified sampling to maintain answerable/unanswerable ratio."""
        # Identify answerable examples
        is_answerable = [
            len(example['answers']['text']) > 0
            for example in dataset
        ]

        answerable_indices = [i for i, ans in enumerate(is_answerable) if ans]
        unanswerable_indices = [i for i, ans in enumerate(is_answerable) if not ans]

        # Calculate how many of each type to sample
        answerable_ratio = len(answerable_indices) / len(dataset)
        n_answerable = int(max_size * answerable_ratio)
        n_unanswerable = max_size - n_answerable

        # Sample indices
        sampled_answerable = np.random.choice(
            answerable_indices,
            min(n_answerable, len(answerable_indices)),
            replace=False
        )
        sampled_unanswerable = np.random.choice(
            unanswerable_indices,
            min(n_unanswerable, len(unanswerable_indices)),
            replace=False
        )

        # Combine and shuffle
        all_indices = np.concatenate([sampled_answerable, sampled_unanswerable])
        np.random.shuffle(all_indices)

        return dataset.select(all_indices)

    def _process_squad_for_combined(self, dataset: Dataset) -> Dict[str, List]:
        """Process SQuAD data for combined dataset format."""
        processed = {
            'question': [],
            'context': [],
            'answers': [],
            'is_answerable': [],
            'source': [],
            'passage_id': [],
            'relevance_score': []
        }

        for i, example in enumerate(dataset):
            processed['question'].append(example['question'])
            processed['context'].append(example['context'])
            processed['answers'].append(example['answers'])
            processed['is_answerable'].append(len(example['answers']['text']) > 0)
            processed['source'].append('squad_v2')
            processed['passage_id'].append(f"squad_{i}")
            # For SQuAD, assume high relevance since context is provided
            processed['relevance_score'].append(1.0)

        return processed

    def _process_marco_for_combined(self, dataset: Dataset) -> Dict[str, List]:
        """Process MS MARCO data for combined dataset format.

        MS MARCO passages field is a dict with keys 'is_selected', 'passage_text', 'url',
        each containing a list of values (one per passage).
        """
        processed = {
            'question': [],
            'context': [],
            'answers': [],
            'is_answerable': [],
            'source': [],
            'passage_id': [],
            'relevance_score': []
        }

        max_passages = self.config.get('data.max_passages_per_question', 10)

        for i, example in enumerate(dataset):
            query = example['query']
            passages = example['passages']

            # MS MARCO passages is a dict of lists, not a list of dicts
            passage_texts = passages.get('passage_text', [])
            is_selected_list = passages.get('is_selected', [])

            # Take top-k passages
            num_passages = min(len(passage_texts), max_passages)
            for j in range(num_passages):
                passage_text = passage_texts[j]
                is_selected = is_selected_list[j] if j < len(is_selected_list) else 0

                processed['question'].append(query)
                processed['context'].append(passage_text)

                # MS MARCO doesn't have explicit answers, treat as answerable
                # if passage is selected (is_selected = 1)
                processed['answers'].append({
                    'text': [passage_text[:100]] if is_selected else [],
                    'answer_start': [0] if is_selected else []
                })
                processed['is_answerable'].append(bool(is_selected))
                processed['source'].append('ms_marco')
                processed['passage_id'].append(f"marco_{i}_{j}")
                processed['relevance_score'].append(float(is_selected))

        return processed

    def _get_dataset_info(self, dataset: Union[Dataset, DatasetDict]) -> str:
        """Get summary information about a dataset."""
        if isinstance(dataset, DatasetDict):
            info_parts = []
            for split, data in dataset.items():
                info_parts.append(f"{split}={len(data)}")
            return ", ".join(info_parts)
        else:
            return f"{len(dataset)} examples"

    def get_dataset_statistics(self, dataset: Dataset) -> Dict[str, Union[int, float]]:
        """Calculate statistics for a combined dataset.

        Args:
            dataset: Combined dataset to analyze.

        Returns:
            Dictionary containing dataset statistics.
        """
        stats = {
            'total_examples': len(dataset),
            'answerable_count': sum(dataset['is_answerable']),
            'unanswerable_count': sum(1 - ans for ans in dataset['is_answerable']),
            'squad_examples': sum(1 for source in dataset['source'] if source == 'squad_v2'),
            'marco_examples': sum(1 for source in dataset['source'] if source == 'ms_marco'),
        }

        stats['answerable_ratio'] = stats['answerable_count'] / stats['total_examples']
        stats['avg_question_length'] = np.mean([len(q.split()) for q in dataset['question']])
        stats['avg_context_length'] = np.mean([len(c.split()) for c in dataset['context']])

        if 'relevance_score' in dataset.column_names:
            relevance_scores = [score for score in dataset['relevance_score'] if score is not None]
            if relevance_scores:
                stats['avg_relevance_score'] = np.mean(relevance_scores)
                stats['min_relevance_score'] = np.min(relevance_scores)
                stats['max_relevance_score'] = np.max(relevance_scores)

        return stats