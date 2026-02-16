"""Adaptive Retrieval QA with Answerability Calibration.

A retrieval-augmented question answering system that learns when to abstain
from answering by combining MS MARCO passage retrieval with SQuAD 2.0's
unanswerable question detection.
"""

__version__ = "0.1.0"
__author__ = "ML Research Team"

from .models.model import AdaptiveRetrievalQAModel
from .training.trainer import AdaptiveQATrainer
from .evaluation.metrics import AnswerabilityCalibrationMetrics

__all__ = [
    "AdaptiveRetrievalQAModel",
    "AdaptiveQATrainer",
    "AnswerabilityCalibrationMetrics",
]