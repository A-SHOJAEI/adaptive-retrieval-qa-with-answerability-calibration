"""Comprehensive evaluation metrics for retrieval-augmented QA with answerability calibration.

This module implements advanced evaluation metrics including exact match, F1 score,
answerability AUROC, retrieval MRR@10, and Expected Calibration Error (ECE).
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config import Config


class AnswerabilityCalibrationMetrics:
    """Comprehensive metrics for evaluating answerability prediction and calibration.

    This class implements all key metrics for the adaptive retrieval QA system,
    including traditional QA metrics and novel calibration metrics.
    """

    def __init__(self, config: Config) -> None:
        """Initialize metrics calculator.

        Args:
            config: Configuration object containing metric parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Metric configuration
        self.confidence_threshold = config.get('model.confidence_threshold', 0.5)
        self.calibration_bins = config.get('calibration.confidence_bins', 10)

        # Target metrics for comparison
        self.target_metrics = config.get('evaluation.target_metrics', {})

    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """Comprehensive model evaluation.

        Args:
            model: The model to evaluate.
            dataloader: DataLoader with evaluation data.
            device: Device to run evaluation on.
            return_predictions: Whether to return raw predictions.

        Returns:
            Dictionary containing all evaluation metrics.
        """
        self.logger.info("Starting comprehensive model evaluation")

        model.eval()
        all_predictions = []
        all_ground_truths = []
        all_answerability_preds = []
        all_answerability_probs = []
        all_answerability_labels = []
        all_retrieval_scores = []
        all_qa_confidences = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)

                # Extract QA predictions
                start_logits = outputs['start_logits']
                end_logits = outputs['end_logits']
                start_probs = F.softmax(start_logits, dim=-1)
                end_probs = F.softmax(end_logits, dim=-1)

                # Get answer spans
                start_preds = torch.argmax(start_logits, dim=-1)
                end_preds = torch.argmax(end_logits, dim=-1)

                # Extract ground truth
                start_labels = batch.get('start_positions')
                end_labels = batch.get('end_positions')

                # QA confidence (max probability product)
                qa_conf = []
                for i in range(start_probs.size(0)):
                    max_start_prob = torch.max(start_probs[i])
                    max_end_prob = torch.max(end_probs[i])
                    qa_conf.append((max_start_prob * max_end_prob).item())

                # Answerability predictions
                answerability_probs = outputs['answerability_probs']
                answerability_preds = answerability_probs > self.confidence_threshold

                # Collect all data
                if start_labels is not None and end_labels is not None:
                    for i in range(len(start_labels)):
                        # QA exact match
                        exact_match = (
                            start_preds[i].item() == start_labels[i].item() and
                            end_preds[i].item() == end_labels[i].item()
                        )
                        all_predictions.append(exact_match)
                        all_ground_truths.append(True)  # All have ground truth

                all_answerability_probs.extend(answerability_probs.cpu().numpy())
                all_answerability_preds.extend(answerability_preds.cpu().numpy())
                all_answerability_labels.extend(batch['is_answerable'].cpu().numpy())
                all_qa_confidences.extend(qa_conf)

                # Retrieval scores if available
                if 'retrieval_scores' in batch:
                    all_retrieval_scores.extend(batch['retrieval_scores'].cpu().numpy())

        # Calculate all metrics
        metrics = {}

        # 1. Exact Match and F1 Score
        if all_predictions:
            metrics.update(self._calculate_qa_metrics(all_predictions, all_ground_truths))

        # 2. Answerability AUROC
        if all_answerability_labels and all_answerability_probs:
            metrics.update(self._calculate_answerability_metrics(
                all_answerability_labels,
                all_answerability_probs,
                all_answerability_preds
            ))

        # 3. Retrieval MRR@10
        if all_retrieval_scores:
            metrics.update(self._calculate_retrieval_metrics(all_retrieval_scores))

        # 4. Calibration ECE
        if all_answerability_probs and all_answerability_labels:
            metrics.update(self._calculate_calibration_metrics(
                all_answerability_probs,
                all_answerability_labels
            ))

        # 5. Additional analysis metrics
        metrics.update(self._calculate_additional_metrics(
            all_answerability_probs,
            all_answerability_labels,
            all_qa_confidences
        ))

        # Compare with targets
        metrics.update(self._compare_with_targets(metrics))

        self.logger.info("Evaluation completed")
        self._log_metrics_summary(metrics)

        if return_predictions:
            predictions = {
                'qa_predictions': all_predictions,
                'answerability_probs': all_answerability_probs,
                'answerability_preds': all_answerability_preds,
                'answerability_labels': all_answerability_labels,
                'qa_confidences': all_qa_confidences,
                'retrieval_scores': all_retrieval_scores
            }
            return metrics, predictions

        return metrics

    def _calculate_qa_metrics(
        self,
        predictions: List[bool],
        ground_truths: List[bool]
    ) -> Dict[str, float]:
        """Calculate QA metrics (exact match, F1).

        Args:
            predictions: List of exact match predictions.
            ground_truths: List of ground truth values.

        Returns:
            Dictionary with QA metrics.
        """
        exact_match = np.mean(predictions)

        # For F1, treat each exact match as binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truths,
            predictions,
            average='binary',
            zero_division=0
        )

        return {
            'exact_match': exact_match,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

    def _calculate_answerability_metrics(
        self,
        labels: List[float],
        probabilities: List[float],
        predictions: List[bool]
    ) -> Dict[str, float]:
        """Calculate answerability prediction metrics.

        Args:
            labels: True answerability labels.
            probabilities: Predicted probabilities.
            predictions: Binary predictions.

        Returns:
            Dictionary with answerability metrics.
        """
        labels_np = np.array(labels)
        probs_np = np.array(probabilities)
        preds_np = np.array(predictions, dtype=int)

        # AUROC
        try:
            auroc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            auroc = 0.5  # Random baseline if only one class

        # Average Precision
        try:
            avg_precision = average_precision_score(labels_np, probs_np)
        except ValueError:
            avg_precision = np.mean(labels_np)

        # Accuracy, Precision, Recall, F1
        accuracy = accuracy_score(labels_np, preds_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np,
            preds_np,
            average='binary',
            zero_division=0
        )

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(labels_np, preds_np).ravel()

        return {
            'answerability_auroc': auroc,
            'answerability_avg_precision': avg_precision,
            'answerability_accuracy': accuracy,
            'answerability_precision': precision,
            'answerability_recall': recall,
            'answerability_f1': f1,
            'answerability_true_positives': int(tp),
            'answerability_false_positives': int(fp),
            'answerability_true_negatives': int(tn),
            'answerability_false_negatives': int(fn)
        }

    def _calculate_retrieval_metrics(
        self,
        retrieval_scores: List[float]
    ) -> Dict[str, float]:
        """Calculate retrieval metrics (MRR@10).

        Args:
            retrieval_scores: List of retrieval relevance scores.

        Returns:
            Dictionary with retrieval metrics.
        """
        # For simplicity, assume retrieval scores are ordered by relevance
        # In practice, this would need the ranking information

        scores_np = np.array(retrieval_scores)

        # Mean Reciprocal Rank approximation
        # Find position of first relevant score (score > threshold)
        threshold = 0.5
        mrr_scores = []

        # Group scores by query (assuming top_k consecutive scores per query)
        top_k = self.config.get('model.retrieval_top_k', 10)
        num_queries = len(scores_np) // top_k

        for i in range(0, len(scores_np), top_k):
            query_scores = scores_np[i:i+top_k]
            # Find rank of first relevant passage
            relevant_indices = np.where(query_scores > threshold)[0]
            if len(relevant_indices) > 0:
                rank = relevant_indices[0] + 1  # 1-indexed rank
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)

        mrr_at_10 = np.mean(mrr_scores) if mrr_scores else 0.0

        return {
            'retrieval_mrr@10': mrr_at_10,
            'retrieval_mean_score': np.mean(scores_np),
            'retrieval_std_score': np.std(scores_np),
            'retrieval_coverage': np.mean(scores_np > threshold)
        }

    def _calculate_calibration_metrics(
        self,
        probabilities: List[float],
        labels: List[float]
    ) -> Dict[str, float]:
        """Calculate calibration metrics (ECE, MCE, etc.).

        Args:
            probabilities: Predicted probabilities.
            labels: True binary labels.

        Returns:
            Dictionary with calibration metrics.
        """
        probs_np = np.array(probabilities)
        labels_np = np.array(labels)

        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(probs_np, labels_np)

        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(probs_np, labels_np)

        # Brier Score
        brier_score = np.mean((probs_np - labels_np) ** 2)

        # Reliability diagram data
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs_np > bin_lower) & (probs_np <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = labels_np[in_bin].mean()
                avg_confidence_in_bin = probs_np[in_bin].mean()
                count_in_bin = in_bin.sum()
            else:
                accuracy_in_bin = 0
                avg_confidence_in_bin = 0
                count_in_bin = 0

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(int(count_in_bin))

        return {
            'calibration_ece': ece,
            'calibration_mce': mce,
            'brier_score': brier_score,
            'calibration_bin_accuracies': bin_accuracies,
            'calibration_bin_confidences': bin_confidences,
            'calibration_bin_counts': bin_counts
        }

    def _calculate_ece(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate Expected Calibration Error.

        Args:
            probabilities: Predicted probabilities [0, 1].
            labels: True binary labels {0, 1}.

        Returns:
            ECE value.
        """
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Filter predictions in this bin
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = labels[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = probabilities[in_bin].mean()
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _calculate_mce(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate Maximum Calibration Error.

        Args:
            probabilities: Predicted probabilities [0, 1].
            labels: True binary labels {0, 1}.

        Returns:
            MCE value.
        """
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        calibration_errors = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                calibration_errors.append(abs(avg_confidence_in_bin - accuracy_in_bin))

        return max(calibration_errors) if calibration_errors else 0.0

    def _calculate_additional_metrics(
        self,
        answerability_probs: List[float],
        answerability_labels: List[float],
        qa_confidences: List[float]
    ) -> Dict[str, float]:
        """Calculate additional analysis metrics.

        Args:
            answerability_probs: Answerability probabilities.
            answerability_labels: True answerability labels.
            qa_confidences: QA model confidence scores.

        Returns:
            Dictionary with additional metrics.
        """
        metrics = {}

        # Confidence distribution statistics
        ans_probs = np.array(answerability_probs)
        qa_confs = np.array(qa_confidences)

        metrics.update({
            'confidence_mean': float(np.mean(ans_probs)),
            'confidence_std': float(np.std(ans_probs)),
            'confidence_min': float(np.min(ans_probs)),
            'confidence_max': float(np.max(ans_probs)),
            'qa_confidence_mean': float(np.mean(qa_confs)),
            'qa_confidence_std': float(np.std(qa_confs))
        })

        # Correlation between answerability and QA confidence
        if len(qa_confs) > 1:
            correlation = np.corrcoef(ans_probs, qa_confs)[0, 1]
            metrics['answerability_qa_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0

        # Coverage at different confidence levels
        confidence_thresholds = [0.3, 0.5, 0.7, 0.9]
        for thresh in confidence_thresholds:
            coverage = np.mean(ans_probs >= thresh)
            accuracy_at_thresh = np.mean(
                np.array(answerability_labels)[ans_probs >= thresh]
            ) if coverage > 0 else 0.0

            metrics[f'coverage@{thresh}'] = float(coverage)
            metrics[f'accuracy@{thresh}'] = float(accuracy_at_thresh)

        return metrics

    def _compare_with_targets(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Compare metrics with target values.

        Args:
            metrics: Current metrics.

        Returns:
            Dictionary with target comparisons.
        """
        comparisons = {}

        for metric_name, target_value in self.target_metrics.items():
            if metric_name in metrics:
                current_value = metrics[metric_name]
                # Calculate relative difference
                if target_value != 0:
                    relative_diff = (current_value - target_value) / target_value
                else:
                    relative_diff = current_value

                comparisons[f'{metric_name}_vs_target'] = relative_diff
                comparisons[f'{metric_name}_target_achieved'] = float(current_value >= target_value)

        # Overall target achievement rate
        achieved_targets = [v for k, v in comparisons.items() if k.endswith('_target_achieved')]
        if achieved_targets:
            comparisons['overall_target_achievement'] = np.mean(achieved_targets)

        return comparisons

    def _log_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """Log summary of key metrics.

        Args:
            metrics: Dictionary of all metrics.
        """
        key_metrics = [
            'exact_match', 'f1_score', 'answerability_auroc',
            'retrieval_mrr@10', 'calibration_ece'
        ]

        self.logger.info("=== Evaluation Summary ===")
        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                target = self.target_metrics.get(metric)
                if target is not None:
                    status = "✓" if value >= target else "✗"
                    self.logger.info(f"{metric}: {value:.4f} (target: {target:.4f}) {status}")
                else:
                    self.logger.info(f"{metric}: {value:.4f}")

    def plot_calibration_curve(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curve (reliability diagram).

        Args:
            probabilities: Predicted probabilities.
            labels: True binary labels.
            save_path: Optional path to save the plot.
        """
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probabilities, n_bins=self.calibration_bins
        )

        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            'bo-',
            label='Model calibration'
        )

        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve (Reliability Diagram)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        # Add ECE to plot
        ece = self._calculate_ece(probabilities, labels)
        plt.text(0.05, 0.95, f'ECE = {ece:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Calibration curve saved to {save_path}")

        plt.close()

    def plot_confidence_histogram(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confidence histogram split by true/false labels.

        Args:
            probabilities: Predicted probabilities.
            labels: True binary labels.
            save_path: Optional path to save the plot.
        """
        plt.figure(figsize=(10, 6))

        # Separate by labels
        true_probs = probabilities[labels == 1]
        false_probs = probabilities[labels == 0]

        plt.hist(false_probs, bins=20, alpha=0.5, label='Unanswerable', color='red', density=True)
        plt.hist(true_probs, bins=20, alpha=0.5, label='Answerable', color='green', density=True)

        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Confidence Distribution by True Label')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add threshold line
        plt.axvline(self.confidence_threshold, color='black', linestyle='--',
                   label=f'Threshold ({self.confidence_threshold})')
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confidence histogram saved to {save_path}")

        plt.close()

    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
        save_dir: str
    ) -> str:
        """Generate comprehensive evaluation report.

        Args:
            metrics: Dictionary of all metrics.
            save_dir: Directory to save report files.

        Returns:
            Path to the generated report file.
        """
        import os
        from pathlib import Path

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        report_path = save_path / "evaluation_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Adaptive Retrieval QA - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")

            # Key metrics
            f.write("Key Metrics:\n")
            f.write("-" * 12 + "\n")
            key_metrics = [
                ('Exact Match', 'exact_match'),
                ('F1 Score', 'f1_score'),
                ('Answerability AUROC', 'answerability_auroc'),
                ('Retrieval MRR@10', 'retrieval_mrr@10'),
                ('Calibration ECE', 'calibration_ece')
            ]

            for display_name, metric_key in key_metrics:
                if metric_key in metrics:
                    value = metrics[metric_key]
                    target = self.target_metrics.get(metric_key)
                    if target is not None:
                        status = "ACHIEVED" if value >= target else "NOT ACHIEVED"
                        f.write(f"{display_name}: {value:.4f} (target: {target:.4f}) - {status}\n")
                    else:
                        f.write(f"{display_name}: {value:.4f}\n")

            # Detailed metrics
            f.write(f"\nDetailed Metrics:\n")
            f.write("-" * 17 + "\n")
            for metric, value in sorted(metrics.items()):
                if isinstance(value, (int, float)) and not metric.endswith('_counts'):
                    f.write(f"{metric}: {value:.4f}\n")

            # Target achievement summary
            if self.target_metrics:
                achieved = sum(1 for k in self.target_metrics.keys()
                             if k in metrics and metrics[k] >= self.target_metrics[k])
                total = len(self.target_metrics)
                f.write(f"\nTarget Achievement: {achieved}/{total} ({100*achieved/total:.1f}%)\n")

        self.logger.info(f"Evaluation report saved to {report_path}")
        return str(report_path)