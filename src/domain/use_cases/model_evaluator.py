"""
ModelEvaluator — Comprehensive model evaluation and testing on held-out test set.

Implements:
- Per-model testing on test set
- Detailed metrics calculation (F1, Accuracy, Precision, Recall, AUC, etc.)
- Confusion matrix generation
- Classification reports
- Performance comparison across models
"""

import logging
from typing import Any, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
)
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates models on held-out test set."""

    def __init__(self, output_dir: str = "output/latest_run/test_results"):
        """
        Initialize ModelEvaluator.

        Args:
            output_dir: Directory to save test results and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results: Dict[str, Dict[str, Any]] = {}

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_name: str,
        class_labels: Optional[np.ndarray] = None,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on the test set.

        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test labels (ground truth)
            model_name: Name of the model
            class_labels: Optional class label names
            y_pred_proba: Optional predicted probabilities (for AUC calculation)

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING: {model_name.upper()}")
        logger.info(f"{'='*70}")

        # Generate predictions
        y_pred = model.predict(X_test)

        # Validate predictions
        if len(y_pred) != len(y_test):
            raise ValueError(
                f"Prediction mismatch for {model_name}: "
                f"got {len(y_pred)} predictions but expected {len(y_test)}"
            )

        # Calculate metrics
        metrics = {
            "model_name": model_name,
            "n_test_samples": len(y_test),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        # Per-class metrics
        per_class_metrics = self._compute_per_class_metrics(
            y_test, y_pred, class_labels
        )
        metrics["per_class"] = per_class_metrics

        # Calculate log loss if probabilities available
        if y_pred_proba is not None:
            try:
                metrics["log_loss"] = log_loss(y_test, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate log loss: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        metrics["classification_report"] = report

        # Store results
        self.test_results[model_name] = metrics

        # Log results
        self._log_test_results(metrics, class_labels)

        return metrics

    def _compute_per_class_metrics(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        class_labels: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-class metrics."""
        per_class = {}

        for class_idx in np.unique(np.concatenate([y_test, y_pred])):
            binary_true = (y_test == class_idx).astype(int)
            binary_pred = (y_pred == class_idx).astype(int)

            class_label = class_labels[class_idx] if class_labels is not None else f"Class {class_idx}"

            per_class[str(class_idx)] = {
                "label": class_label,
                "accuracy": accuracy_score(binary_true, binary_pred),
                "f1": f1_score(binary_true, binary_pred, zero_division=0),
                "precision": precision_score(binary_true, binary_pred, zero_division=0),
                "recall": recall_score(binary_true, binary_pred, zero_division=0),
                "support": int(np.sum(y_test == class_idx)),
            }

        return per_class

    def _log_test_results(
        self,
        metrics: Dict[str, Any],
        class_labels: Optional[np.ndarray] = None,
    ):
        """Log test results in human-readable format."""
        logger.info(f"\n📊 Test Set Performance ({metrics['n_test_samples']} samples):")
        logger.info(f"   Accuracy:          {metrics['accuracy']:.4f}")
        logger.info(f"   F1-Macro:          {metrics['f1_macro']:.4f}")
        logger.info(f"   F1-Weighted:       {metrics['f1_weighted']:.4f}")
        logger.info(f"   Precision (macro): {metrics['precision_macro']:.4f}")
        logger.info(f"   Recall (macro):    {metrics['recall_macro']:.4f}")

        if "log_loss" in metrics:
            logger.info(f"   Log Loss:          {metrics['log_loss']:.4f}")

        logger.info(f"\n📋 Per-Class Performance:")
        for class_idx, class_metrics in sorted(metrics.get("per_class", {}).items()):
            label = class_metrics.get("label", f"Class {class_idx}")
            logger.info(f"   {label}:")
            logger.info(f"      F1: {class_metrics['f1']:.4f} | "
                       f"Precision: {class_metrics['precision']:.4f} | "
                       f"Recall: {class_metrics['recall']:.4f} | "
                       f"Support: {class_metrics['support']}")

        logger.info(f"\n🔗 Confusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        logger.info("\n" + self._format_confusion_matrix(cm, class_labels))

    def _format_confusion_matrix(
        self,
        cm: np.ndarray,
        class_labels: Optional[np.ndarray] = None,
    ) -> str:
        """Format confusion matrix for logging."""
        n_classes = cm.shape[0]
        labels = (
            class_labels if class_labels is not None
            else np.array([f"C{i}" for i in range(n_classes)])
        )

        # Create header
        lines = ["   Predicted →"]
        header = "   " + " ".join([f"{l:>8}" for l in labels])
        lines.append(header)

        # Add rows
        for i, label in enumerate(labels):
            row_label = f"{label:>3}"
            row_values = " ".join([f"{cm[i, j]:>8}" for j in range(n_classes)])
            lines.append(f"{row_label} {row_values}")

        return "\n".join(lines)

    def compare_models(self) -> pd.DataFrame:
        """
        Create a comparison DataFrame for all evaluated models.

        Returns:
            DataFrame with model comparison
        """
        if not self.test_results:
            logger.warning("No test results to compare")
            return pd.DataFrame()

        comparison_data = []

        for model_name, metrics in self.test_results.items():
            comparison_data.append({
                "Model": model_name,
                "Accuracy": metrics.get("accuracy"),
                "F1-Macro": metrics.get("f1_macro"),
                "F1-Weighted": metrics.get("f1_weighted"),
                "Precision-Macro": metrics.get("precision_macro"),
                "Recall-Macro": metrics.get("recall_macro"),
                "Log Loss": metrics.get("log_loss"),
            })

        df_comparison = pd.DataFrame(comparison_data).sort_values("F1-Macro", ascending=False)

        logger.info(f"\n{'='*70}")
        logger.info("MODEL COMPARISON ON TEST SET")
        logger.info(f"{'='*70}")
        logger.info("\n" + df_comparison.to_string(index=False))

        return df_comparison

    def save_test_results(self) -> Path:
        """
        Save test results to JSON and CSV files.

        Returns:
            Path to the results directory
        """
        # Save detailed results as JSON
        results_json = self.output_dir / "test_results.json"
        with open(results_json, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        logger.info(f"✓ Saved detailed results to {results_json}")

        # Save comparison as CSV
        df_comparison = self.compare_models()
        if not df_comparison.empty:
            comparison_csv = self.output_dir / "model_comparison.csv"
            df_comparison.to_csv(comparison_csv, index=False)
            logger.info(f"✓ Saved model comparison to {comparison_csv}")

        # Save confusion matrices
        cm_dir = self.output_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)

        for model_name, metrics in self.test_results.items():
            cm = np.array(metrics["confusion_matrix"])
            cm_file = cm_dir / f"{model_name}_confusion_matrix.txt"

            with open(cm_file, "w") as f:
                f.write(f"Confusion Matrix for {model_name}\n")
                f.write("=" * 50 + "\n")
                f.write(str(cm) + "\n")

            logger.info(f"✓ Saved confusion matrix to {cm_file}")

        logger.info(f"✅ All test results saved to {self.output_dir}")
        return self.output_dir

    def generate_test_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate a comprehensive test report.

        Args:
            output_file: Optional path to save the report as text

        Returns:
            Report text
        """
        report_lines = [
            "=" * 70,
            "TEST RESULTS REPORT",
            "=" * 70,
            "",
        ]

        if not self.test_results:
            report_lines.append("No test results available")
            report_text = "\n".join(report_lines)

            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as f:
                    f.write(report_text)

            return report_text

        # Add model comparison
        df_comparison = self.compare_models()
        report_lines.extend([
            "MODEL PERFORMANCE COMPARISON",
            "-" * 70,
            df_comparison.to_string(index=False),
            "",
        ])

        # Add detailed results per model
        for model_name, metrics in sorted(self.test_results.items()):
            report_lines.extend([
                f"DETAILED RESULTS: {model_name}",
                "-" * 70,
                f"Test Set Size: {metrics['n_test_samples']} samples",
                f"Accuracy:      {metrics['accuracy']:.4f}",
                f"F1-Macro:      {metrics['f1_macro']:.4f}",
                f"F1-Weighted:   {metrics['f1_weighted']:.4f}",
                f"Precision:     {metrics['precision_macro']:.4f}",
                f"Recall:        {metrics['recall_macro']:.4f}",
            ])

            if "log_loss" in metrics:
                report_lines.append(f"Log Loss:      {metrics['log_loss']:.4f}")

            report_lines.append("")

        report_text = "\n".join(report_lines)

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report_text)
            logger.info(f"✓ Test report saved to {output_file}")

        return report_text
