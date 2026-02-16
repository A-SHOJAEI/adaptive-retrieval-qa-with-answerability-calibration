"""Data preprocessing utilities for retrieval-augmented QA."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from sentence_transformers import SentenceTransformer

from ..utils.config import Config


# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to download NLTK data: {e}. Some features may not work properly.")


class DataPreprocessor:
    """Preprocesses data for the adaptive retrieval QA system.

    This class handles tokenization, passage chunking, feature extraction,
    and preparation of training examples for both retrieval and QA components.
    """

    def __init__(
        self,
        config: Config,
        tokenizer: Optional[AutoTokenizer] = None,
        retriever: Optional[SentenceTransformer] = None
    ) -> None:
        """Initialize data preprocessor.

        Args:
            config: Configuration object.
            tokenizer: Tokenizer for the QA model. If None, loads from config.
            retriever: Sentence transformer for passage encoding. If None, loads from config.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize tokenizer
        if tokenizer is None:
            reader_name = config.get('model.reader_name', 'deepset/roberta-base-squad2')
            self.tokenizer = AutoTokenizer.from_pretrained(reader_name)
        else:
            self.tokenizer = tokenizer

        # Initialize retriever for encoding
        if retriever is None:
            retriever_name = config.get(
                'model.retriever_name',
                'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
            )
            self.retriever = SentenceTransformer(retriever_name)
        else:
            self.retriever = retriever

        self.max_seq_length = config.get('model.max_seq_length', 512)
        self.passage_max_length = config.get('data.passage_max_length', 200)
        self.answer_max_length = config.get('model.answer_max_length', 100)

    def preprocess_dataset(
        self,
        dataset: Dataset,
        is_training: bool = True
    ) -> Dataset:
        """Preprocess a complete dataset.

        Args:
            dataset: Input dataset to preprocess.
            is_training: Whether this is training data (affects augmentation).

        Returns:
            Preprocessed dataset with additional features.
        """
        self.logger.info(f"Preprocessing dataset with {len(dataset)} examples")

        # Clean and normalize text
        dataset = dataset.map(
            self._clean_text,
            batched=True,
            desc="Cleaning text"
        )

        # Chunk long passages if needed
        dataset = dataset.map(
            self._chunk_passages,
            desc="Chunking passages"
        )

        # Extract confidence features BEFORE tokenization (needs question/context)
        dataset = dataset.map(
            self._extract_confidence_features,
            desc="Extracting confidence features"
        )

        # Tokenize for QA model (removes original text columns)
        # Keep confidence_features and relevance_score through tokenization
        cols_to_keep = {'confidence_features', 'relevance_score'}
        cols_to_remove = [c for c in dataset.column_names
                          if c not in cols_to_keep]
        dataset = dataset.map(
            lambda examples: self._tokenize_examples(examples, is_training),
            batched=True,
            remove_columns=cols_to_remove,
            desc="Tokenizing examples"
        )

        self.logger.info("Dataset preprocessing completed")
        return dataset

    def _clean_text(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Clean and normalize text data.

        Args:
            examples: Batch of examples to clean.

        Returns:
            Cleaned examples.
        """
        cleaned_questions = []
        cleaned_contexts = []

        for question, context in zip(examples['question'], examples['context']):
            # Clean question
            clean_q = self._normalize_text(question)
            cleaned_questions.append(clean_q)

            # Clean context
            clean_c = self._normalize_text(context)
            cleaned_contexts.append(clean_c)

        examples['question'] = cleaned_questions
        examples['context'] = cleaned_contexts
        return examples

    def _normalize_text(self, text: str) -> str:
        """Normalize a single text string.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()\-]', '', text)

        # Normalize quotes using unicode escape sequences
        text = re.sub(r'[\u201c\u201d\u201f]', '"', text)  # Smart double quotes
        text = re.sub(r'[\u2018\u2019\u201b]', "'", text)  # Smart single quotes

        return text.strip()

    def _chunk_passages(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk long passages into smaller segments.

        Args:
            example: Single example to process.

        Returns:
            Example with chunked passage if necessary.
        """
        context = example['context']

        # If passage is short enough, return as-is
        words = context.split()
        if len(words) <= self.passage_max_length:
            return example

        # Split into chunks with overlap
        chunk_size = self.passage_max_length
        overlap = chunk_size // 4  # 25% overlap

        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

            # If this chunk goes to the end, break
            if i + chunk_size >= len(words):
                break

        # For training, try to find the chunk containing the answer
        if 'answers' in example and example['answers']['answer_start']:
            answer_start = example['answers']['answer_start'][0]
            best_chunk = context  # Default fallback

            # Find chunk that best contains the answer
            for chunk in chunks:
                if chunk in context:
                    chunk_start = context.find(chunk)
                    chunk_end = chunk_start + len(chunk)
                    if chunk_start <= answer_start <= chunk_end:
                        best_chunk = chunk
                        # Update answer start position relative to chunk
                        example['answers']['answer_start'] = [answer_start - chunk_start]
                        break

            example['context'] = best_chunk
        else:
            # For non-training or unanswerable questions, use first chunk
            example['context'] = chunks[0] if chunks else context

        return example

    def _tokenize_examples(
        self,
        examples: Dict[str, List],
        is_training: bool
    ) -> Dict[str, List]:
        """Tokenize examples for QA model.

        Args:
            examples: Batch of examples to tokenize.
            is_training: Whether this is training data.

        Returns:
            Tokenized examples.
        """
        questions = examples['question']
        contexts = examples['context']

        # Tokenize question-context pairs
        tokenized = self.tokenizer(
            questions,
            contexts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        # Prepare outputs
        result = {
            'input_ids': tokenized['input_ids'].tolist(),
            'attention_mask': tokenized['attention_mask'].tolist(),
            'token_type_ids': tokenized.get('token_type_ids', torch.zeros_like(tokenized['input_ids'])).tolist(),
            'is_answerable': examples.get('is_answerable', [True] * len(questions)),
            'source': examples.get('source', ['unknown'] * len(questions)),
        }

        # Add answer positions for training
        if is_training and 'answers' in examples:
            start_positions = []
            end_positions = []

            for i, (answer, offset_mapping) in enumerate(
                zip(examples['answers'], tokenized['offset_mapping'])
            ):
                if not answer['answer_start'] or not answer['text']:
                    # Unanswerable question
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Find answer span in tokens
                    answer_start = answer['answer_start'][0]
                    answer_text = answer['text'][0]
                    answer_end = answer_start + len(answer_text)

                    # Find token positions
                    start_pos = 0
                    end_pos = 0

                    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                        if char_start <= answer_start < char_end:
                            start_pos = token_idx
                        if char_start < answer_end <= char_end:
                            end_pos = token_idx
                            break

                    start_positions.append(start_pos)
                    end_positions.append(end_pos)

            result['start_positions'] = start_positions
            result['end_positions'] = end_positions

        return result

    def _encode_passages(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Encode passages using sentence transformer.

        Args:
            examples: Batch of examples with contexts.

        Returns:
            Examples with passage embeddings.
        """
        contexts = examples.get('context', [])
        if not contexts:
            return examples

        # Encode passages
        embeddings = self.retriever.encode(
            contexts,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        examples['passage_embeddings'] = embeddings.tolist()
        return examples

    def _extract_confidence_features(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for confidence calibration.

        Args:
            example: Single example to process.

        Returns:
            Example with confidence features.
        """
        # Question-level features
        question = example.get('question', '')
        context = example.get('context', '')

        features = {
            'question_length': len(question.split()),
            'context_length': len(context.split()),
            'question_word_overlap': self._calculate_word_overlap(question, context),
            'has_question_words': self._has_question_words(question),
            'context_complexity': self._calculate_text_complexity(context),
        }

        # Retrieval features
        if 'relevance_score' in example:
            features['retrieval_score'] = example['relevance_score']
        else:
            features['retrieval_score'] = 0.0

        # Source features
        source = example.get('source', 'unknown')
        features['is_squad'] = 1.0 if source == 'squad_v2' else 0.0
        features['is_marco'] = 1.0 if source == 'ms_marco' else 0.0

        example['confidence_features'] = features
        return example

    def _calculate_word_overlap(self, question: str, context: str) -> float:
        """Calculate word overlap between question and context.

        Args:
            question: Question text.
            context: Context text.

        Returns:
            Overlap ratio (0-1).
        """
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except (ImportError, LookupError) as e:
            self.logger.warning(f"Could not load NLTK stopwords, using empty set: {e}")
            stop_words = set()

        question_words = set(word.lower() for word in question.split()
                           if word.lower() not in stop_words and word.isalpha())
        context_words = set(word.lower() for word in context.split()
                          if word.lower() not in stop_words and word.isalpha())

        if not question_words:
            return 0.0

        overlap = len(question_words.intersection(context_words))
        return overlap / len(question_words)

    def _has_question_words(self, question: str) -> float:
        """Check if question contains question words.

        Args:
            question: Question text.

        Returns:
            1.0 if contains question words, 0.0 otherwise.
        """
        question_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'which',
            'whose', 'whom', 'is', 'are', 'was', 'were', 'do', 'does',
            'did', 'can', 'could', 'will', 'would', 'should', 'might'
        }

        question_lower = question.lower()
        return 1.0 if any(word in question_lower for word in question_words) else 0.0

    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity score.

        Args:
            text: Text to analyze.

        Returns:
            Complexity score (higher = more complex).
        """
        if not text:
            return 0.0

        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?') + 1

        # Simple complexity metrics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = len(words) / sentences if sentences > 0 else 0

        # Normalize to 0-1 range
        complexity = (avg_word_length / 10.0 + avg_sentence_length / 20.0) / 2.0
        return min(complexity, 1.0)

    def create_retrieval_corpus(self, dataset: Dataset) -> Tuple[List[str], np.ndarray]:
        """Create passage corpus for retrieval.

        Args:
            dataset: Dataset containing passages.

        Returns:
            Tuple of (passage_texts, passage_embeddings).
        """
        self.logger.info("Creating retrieval corpus")

        # Extract unique passages
        passages = []
        passage_ids = []
        seen_passages = set()

        for example in dataset:
            context = example['context']
            if context not in seen_passages:
                passages.append(context)
                passage_ids.append(example.get('passage_id', len(passages)))
                seen_passages.add(context)

        # Encode all passages
        self.logger.info(f"Encoding {len(passages)} unique passages")
        embeddings = self.retriever.encode(
            passages,
            convert_to_tensor=False,
            show_progress_bar=True
        )

        self.logger.info("Retrieval corpus created")
        return passages, embeddings

    def prepare_inference_input(
        self,
        question: str,
        contexts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Prepare input for inference.

        Args:
            question: Input question.
            contexts: List of retrieved contexts.

        Returns:
            Tokenized inputs for the model.
        """
        # Tokenize question with each context
        all_inputs = []

        for context in contexts:
            inputs = self.tokenizer(
                question,
                context,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            all_inputs.append(inputs)

        # Stack all inputs
        if all_inputs:
            batched_inputs = {
                'input_ids': torch.cat([inp['input_ids'] for inp in all_inputs], dim=0),
                'attention_mask': torch.cat([inp['attention_mask'] for inp in all_inputs], dim=0),
            }

            if 'token_type_ids' in all_inputs[0]:
                batched_inputs['token_type_ids'] = torch.cat(
                    [inp['token_type_ids'] for inp in all_inputs], dim=0
                )

            return batched_inputs
        else:
            # Empty context case
            return self.tokenizer(
                question,
                "",
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )