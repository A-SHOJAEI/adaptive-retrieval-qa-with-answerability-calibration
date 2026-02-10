"""Adaptive Retrieval QA Model with Confidence Calibration.

This module implements a novel retrieval-augmented question answering system
that jointly models retrieval relevance scores and answer extraction confidence
to predict answerability and reduce hallucination.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer
)
from sentence_transformers import SentenceTransformer
import faiss

from ..utils.config import Config


class ConfidenceCalibrator(nn.Module):
    """Confidence calibration module for answerability prediction.

    This module takes retrieval scores and QA model outputs to predict
    whether a question can be answered given the retrieved passages.
    """

    def __init__(
        self,
        qa_hidden_size: int,
        confidence_features_dim: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ) -> None:
        """Initialize confidence calibrator.

        Args:
            qa_hidden_size: Hidden size of the QA model.
            confidence_features_dim: Dimension of confidence features.
            hidden_dim: Hidden dimension for calibration layers.
            dropout: Dropout rate.
        """
        super().__init__()

        self.qa_hidden_size = qa_hidden_size
        self.confidence_features_dim = confidence_features_dim

        # Input projection layer
        input_dim = qa_hidden_size + confidence_features_dim + 3  # +2 for qa_confidence, +1 for retrieval scores
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Calibration layers
        self.calibration_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Temperature scaling parameter
        self.temperature = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        qa_hidden_states: torch.Tensor,
        retrieval_scores: torch.Tensor,
        confidence_features: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for confidence calibration.

        Args:
            qa_hidden_states: Hidden states from QA model [batch_size, hidden_size].
            retrieval_scores: Retrieval relevance scores [batch_size].
            confidence_features: Additional confidence features [batch_size, feature_dim].
            start_logits: Start position logits [batch_size, seq_len].
            end_logits: End position logits [batch_size, seq_len].

        Returns:
            Tuple of (answerability_logits, calibrated_confidence).
        """
        batch_size = qa_hidden_states.size(0)

        # Extract confidence from start/end logits
        start_confidence = torch.max(F.softmax(start_logits, dim=-1), dim=-1)[0]
        end_confidence = torch.max(F.softmax(end_logits, dim=-1), dim=-1)[0]

        # Combine confidence scores
        qa_confidence = torch.stack([start_confidence, end_confidence], dim=-1)

        # Concatenate all features
        features = torch.cat([
            qa_hidden_states,
            qa_confidence,
            retrieval_scores.unsqueeze(-1),
            confidence_features
        ], dim=-1)

        # Project to hidden dimension
        hidden = self.input_projection(features)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        # Get answerability prediction
        answerability_logits = self.calibration_layers(hidden).squeeze(-1)

        # Apply temperature scaling for calibration
        calibrated_logits = answerability_logits / self.temperature
        calibrated_confidence = torch.sigmoid(calibrated_logits)

        return answerability_logits, calibrated_confidence


class AdaptiveRetrievalQAModel(nn.Module):
    """Adaptive Retrieval QA Model with Confidence Calibration.

    This model combines dense passage retrieval with a question answering model
    and a confidence calibration module to predict answerability and reduce
    hallucination in retrieval-augmented generation.
    """

    def __init__(
        self,
        config: Config,
        qa_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        retriever: Optional[SentenceTransformer] = None
    ) -> None:
        """Initialize the adaptive retrieval QA model.

        Args:
            config: Configuration object.
            qa_model: Pre-trained QA model. If None, loads from config.
            tokenizer: Tokenizer. If None, loads from config.
            retriever: Sentence transformer for retrieval. If None, loads from config.
        """
        super().__init__()

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model parameters
        self.confidence_threshold = config.get('model.confidence_threshold', 0.5)
        self.retrieval_top_k = config.get('model.retrieval_top_k', 10)

        # Initialize components
        self._initialize_qa_model(qa_model, tokenizer)
        self._initialize_retriever(retriever)
        self._initialize_calibrator()

        # Passage corpus for retrieval
        self.passage_corpus: Optional[List[str]] = None
        self.passage_index: Optional[faiss.IndexFlatIP] = None
        self.passage_embeddings: Optional[np.ndarray] = None

    def _initialize_qa_model(
        self,
        qa_model: Optional[PreTrainedModel],
        tokenizer: Optional[PreTrainedTokenizer]
    ) -> None:
        """Initialize the question answering model."""
        reader_name = self.config.get('model.reader_name', 'deepset/roberta-base-squad2')

        if qa_model is None:
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(reader_name)
        else:
            self.qa_model = qa_model

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(reader_name)
        else:
            self.tokenizer = tokenizer

        self.logger.info(f"Initialized QA model: {reader_name}")

    def _initialize_retriever(self, retriever: Optional[SentenceTransformer]) -> None:
        """Initialize the passage retriever."""
        retriever_name = self.config.get(
            'model.retriever_name',
            'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
        )

        if retriever is None:
            self.retriever = SentenceTransformer(retriever_name)
        else:
            self.retriever = retriever

        self.logger.info(f"Initialized retriever: {retriever_name}")

    def _initialize_calibrator(self) -> None:
        """Initialize the confidence calibrator."""
        qa_hidden_size = self.qa_model.config.hidden_size
        confidence_features_dim = 10  # From preprocessing features

        self.calibrator = ConfidenceCalibrator(
            qa_hidden_size=qa_hidden_size,
            confidence_features_dim=confidence_features_dim,
            hidden_dim=256,
            dropout=0.1
        )

        self.logger.info("Initialized confidence calibrator")

    def build_passage_index(
        self,
        passages: List[str],
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """Build FAISS index for fast passage retrieval.

        Args:
            passages: List of passage texts.
            embeddings: Pre-computed embeddings. If None, computes them.
        """
        self.logger.info(f"Building passage index with {len(passages)} passages")

        self.passage_corpus = passages

        if embeddings is None:
            self.logger.info("Computing passage embeddings")
            embeddings = self.retriever.encode(
                passages,
                convert_to_tensor=False,
                show_progress_bar=True
            )

        self.passage_embeddings = embeddings

        # Build FAISS index
        embedding_dim = embeddings.shape[1]
        self.passage_index = faiss.IndexFlatIP(embedding_dim)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.passage_index.add(embeddings.astype(np.float32))

        self.logger.info(f"Built FAISS index with {self.passage_index.ntotal} passages")

    def retrieve_passages(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """Retrieve relevant passages for a query.

        Args:
            query: Input query.
            top_k: Number of passages to retrieve. If None, uses config value.

        Returns:
            Tuple of (retrieved_passages, retrieval_scores).

        Raises:
            RuntimeError: If passage index is not built.
        """
        if self.passage_index is None or self.passage_corpus is None:
            raise RuntimeError("Passage index not built. Call build_passage_index first.")

        if top_k is None:
            top_k = self.retrieval_top_k

        # Encode query
        query_embedding = self.retriever.encode([query], convert_to_tensor=False)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.passage_index.search(
            query_embedding.astype(np.float32),
            top_k
        )

        # Extract results
        retrieved_passages = [self.passage_corpus[idx] for idx in indices[0]]
        retrieval_scores = scores[0].tolist()

        return retrieved_passages, retrieval_scores

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        is_answerable: Optional[torch.Tensor] = None,
        retrieval_scores: Optional[torch.Tensor] = None,
        confidence_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the complete adaptive retrieval QA model.

        Processes input through the QA model and calibrates confidence scores using
        multiple signal sources including QA hidden states, retrieval scores, and
        extracted features for answerability prediction.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs with shape [batch_size, seq_len].
                Contains question-context pairs encoded by tokenizer.
            attention_mask (torch.Tensor): Attention mask with shape [batch_size, seq_len].
                Indicates which tokens should be attended to (1) or ignored (0).
            token_type_ids (torch.Tensor, optional): Token type IDs with shape [batch_size, seq_len].
                Distinguishes question tokens (0) from context tokens (1).
            start_positions (torch.Tensor, optional): Ground truth answer start positions
                with shape [batch_size]. Used during training for span prediction loss.
            end_positions (torch.Tensor, optional): Ground truth answer end positions
                with shape [batch_size]. Used during training for span prediction loss.
            is_answerable (torch.Tensor, optional): Binary answerability labels with shape
                [batch_size]. 1 for answerable questions, 0 for unanswerable.
            retrieval_scores (torch.Tensor, optional): Passage relevance scores from retriever
                with shape [batch_size]. Higher scores indicate better passage-question match.
            confidence_features (torch.Tensor, optional): Extracted confidence features
                with shape [batch_size, feature_dim]. Contains lexical, semantic, and
                structural features for calibration.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'start_logits': Answer span start predictions [batch_size, seq_len]
                - 'end_logits': Answer span end predictions [batch_size, seq_len]
                - 'calibrated_confidence': Calibrated confidence scores [batch_size]
                - 'answerability_logits': Binary answerability predictions [batch_size, 2]
                - 'loss': Combined training loss (if labels provided)
                - 'qa_loss': Span prediction loss component
                - 'calibration_loss': Confidence calibration loss component

        Raises:
            RuntimeError: If model components are not properly initialized.
            ValueError: If input tensor shapes are incompatible.

        Note:
            During training, all label tensors should be provided. During inference,
            only input_ids and attention_mask are required.
        """
        # Input validation
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D tensor, got {input_ids.dim()}D")
        if attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be 2D tensor, got {attention_mask.dim()}D")
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"input_ids and attention_mask shape mismatch: {input_ids.shape} vs {attention_mask.shape}")

        batch_size = input_ids.size(0)
        device = input_ids.device

        self.logger.debug(f"Forward pass: batch_size={batch_size}, seq_len={input_ids.size(1)}")

        # QA model forward pass
        self.logger.debug(f"Running QA model forward pass with training={self.training}")
        qa_outputs = self.qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            output_hidden_states=True
        )

        # Extract hidden states for calibration (use [CLS] token)
        qa_hidden_states = qa_outputs.hidden_states[-1][:, 0, :]  # [batch_size, hidden_size]
        self.logger.debug(f"Extracted QA hidden states: {qa_hidden_states.shape}")

        # Default values for missing inputs
        if retrieval_scores is None:
            retrieval_scores = torch.zeros(batch_size, device=device)
            self.logger.debug("Using default retrieval scores (zeros)")
        else:
            self.logger.debug(f"Using provided retrieval scores: {retrieval_scores.shape}")

        if confidence_features is None:
            confidence_features = torch.zeros(batch_size, 10, device=device)
            self.logger.debug("Using default confidence features (zeros)")
        else:
            self.logger.debug(f"Using provided confidence features: {confidence_features.shape}")

        # Confidence calibration
        self.logger.debug("Running confidence calibration")
        answerability_logits, calibrated_confidence = self.calibrator(
            qa_hidden_states=qa_hidden_states,
            retrieval_scores=retrieval_scores,
            confidence_features=confidence_features,
            start_logits=qa_outputs.start_logits,
            end_logits=qa_outputs.end_logits
        )
        self.logger.debug(f"Calibration outputs - answerability: {answerability_logits.shape}, "
                         f"confidence: {calibrated_confidence.shape}")

        outputs = {
            'start_logits': qa_outputs.start_logits,
            'end_logits': qa_outputs.end_logits,
            'answerability_logits': answerability_logits,
            'answerability_probs': calibrated_confidence,
            'qa_loss': qa_outputs.loss if qa_outputs.loss is not None else None
        }

        # Compute calibration loss if labels are provided
        if is_answerable is not None:
            calibration_loss = F.binary_cross_entropy_with_logits(
                answerability_logits,
                is_answerable.float()
            )
            outputs['calibration_loss'] = calibration_loss

            # Combined loss
            if outputs['qa_loss'] is not None:
                # Weighted combination of QA and calibration losses
                qa_weight = self.config.get('training.qa_loss_weight', 1.0)
                cal_weight = self.config.get('training.calibration_loss_weight', 0.5)

                outputs['loss'] = qa_weight * outputs['qa_loss'] + cal_weight * calibration_loss
            else:
                outputs['loss'] = calibration_loss

        return outputs

    def predict(
        self,
        question: str,
        contexts: Optional[List[str]] = None,
        return_confidence: bool = True
    ) -> Dict[str, Union[str, float, List[str]]]:
        """Predict answer for a question with confidence estimation.

        Args:
            question: Input question.
            contexts: Optional pre-retrieved contexts. If None, retrieves automatically.
            return_confidence: Whether to return confidence scores.

        Returns:
            Dictionary containing prediction results.
        """
        self.eval()

        with torch.no_grad():
            # Retrieve passages if not provided
            if contexts is None:
                contexts, retrieval_scores = self.retrieve_passages(question)
            else:
                retrieval_scores = [1.0] * len(contexts)  # Default high scores

            if not contexts:
                return {
                    'answer': '',
                    'confidence': 0.0,
                    'is_answerable': False,
                    'retrieved_passages': [],
                    'retrieval_scores': []
                }

            # Tokenize inputs
            inputs = self.tokenizer(
                [question] * len(contexts),
                contexts,
                max_length=self.config.get('model.max_seq_length', 512),
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Create dummy features for inference
            batch_size = inputs['input_ids'].size(0)
            retrieval_tensor = torch.tensor(retrieval_scores, device=device, dtype=torch.float)
            confidence_features = torch.zeros(batch_size, 10, device=device)

            # Forward pass
            outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids'),
                retrieval_scores=retrieval_tensor,
                confidence_features=confidence_features
            )

            # Find best answer across all passages
            start_probs = F.softmax(outputs['start_logits'], dim=-1)
            end_probs = F.softmax(outputs['end_logits'], dim=-1)

            best_score = 0.0
            best_answer = ""
            best_passage_idx = 0

            for i in range(batch_size):
                # Find best span in this passage
                for start_idx in range(start_probs.size(-1)):
                    for end_idx in range(start_idx, min(start_idx + self.config.get('model.answer_max_length', 100), start_probs.size(-1))):
                        score = start_probs[i, start_idx] * end_probs[i, end_idx]
                        if score > best_score:
                            best_score = score

                            # Extract answer text
                            input_ids = inputs['input_ids'][i]
                            answer_tokens = input_ids[start_idx:end_idx + 1]
                            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

                            if answer_text.strip():  # Only consider non-empty answers
                                best_answer = answer_text.strip()
                                best_passage_idx = i

            # Get answerability prediction
            answerability_scores = outputs['answerability_probs']
            avg_answerability = torch.mean(answerability_scores).item()

            # Make final decision
            is_answerable = (
                avg_answerability >= self.confidence_threshold and
                best_answer != "" and
                best_score >= 0.01  # Minimum confidence threshold
            )

            result = {
                'answer': best_answer if is_answerable else "",
                'confidence': avg_answerability,
                'is_answerable': is_answerable,
                'qa_confidence': float(best_score),
                'retrieved_passages': contexts,
                'retrieval_scores': retrieval_scores
            }

            if return_confidence:
                result.update({
                    'passage_answerability_scores': answerability_scores.cpu().tolist(),
                    'best_passage_idx': best_passage_idx
                })

            return result

    def save_pretrained(self, save_directory: str) -> None:
        """Save model components to directory.

        Args:
            save_directory: Directory to save model files.
        """
        import os
        from pathlib import Path

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save QA model
        qa_path = save_path / "qa_model"
        self.qa_model.save_pretrained(qa_path)
        self.tokenizer.save_pretrained(qa_path)

        # Save calibrator
        calibrator_path = save_path / "calibrator.pt"
        torch.save(self.calibrator.state_dict(), calibrator_path)

        # Save retriever
        retriever_path = save_path / "retriever"
        self.retriever.save(str(retriever_path))

        # Save passage index if available
        if self.passage_index is not None:
            index_path = save_path / "passage_index.faiss"
            faiss.write_index(self.passage_index, str(index_path))

            # Save passage corpus
            corpus_path = save_path / "passage_corpus.txt"
            with open(corpus_path, 'w', encoding='utf-8') as f:
                for passage in self.passage_corpus:
                    f.write(passage + '\n')

        self.logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_directory: str,
        config: Config
    ) -> 'AdaptiveRetrievalQAModel':
        """Load model from saved directory.

        Args:
            model_directory: Directory containing saved model files.
            config: Configuration object.

        Returns:
            Loaded model instance.
        """
        from pathlib import Path

        model_path = Path(model_directory)

        # Load QA model
        qa_path = model_path / "qa_model"
        qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_path)
        tokenizer = AutoTokenizer.from_pretrained(qa_path)

        # Load retriever
        retriever_path = model_path / "retriever"
        retriever = SentenceTransformer(str(retriever_path))

        # Create model instance
        model = cls(config, qa_model, tokenizer, retriever)

        # Load calibrator
        calibrator_path = model_path / "calibrator.pt"
        if calibrator_path.exists():
            model.calibrator.load_state_dict(torch.load(calibrator_path, map_location='cpu'))

        # Load passage index
        index_path = model_path / "passage_index.faiss"
        corpus_path = model_path / "passage_corpus.txt"

        if index_path.exists() and corpus_path.exists():
            model.passage_index = faiss.read_index(str(index_path))

            with open(corpus_path, 'r', encoding='utf-8') as f:
                model.passage_corpus = [line.strip() for line in f]

        return model