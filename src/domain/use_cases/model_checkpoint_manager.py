"""
ModelCheckpointManager — Handles saving and loading model checkpoints with metadata.

Implements:
- Checkpoint saving to timestamped directories
- Per-model-type directory structure
- Metadata tracking (metrics, hyperparams, feature info)
- Model versioning and recovery
"""

import logging
import json
import pickle
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelCheckpointManager:
    """Manages model checkpoints, metrics, and metadata."""

    def __init__(self, base_checkpoint_dir: str = "models/checkpoints"):
        """
        Initialize ModelCheckpointManager.

        Args:
            base_checkpoint_dir: Base directory for all checkpoints
        """
        self.base_dir = Path(base_checkpoint_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_model_checkpoint_dir(self, model_name: str, create: bool = True) -> Path:
        """
        Get the checkpoint directory for a specific model.

        Args:
            model_name: Name of the model (e.g., 'xgboost', 'random_forest', 'svm')
            create: Whether to create the directory if it doesn't exist

        Returns:
            Path to the model-specific checkpoint directory
        """
        checkpoint_dir = self.base_dir / self.timestamp / model_name
        if create:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def save_model_checkpoint(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        cv_results: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        train_stats: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a trained model with its metadata and metrics.

        Args:
            model: Trained model object
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            cv_results: Cross-validation results per fold
            feature_importance: Feature importance DataFrame
            hyperparams: Model hyperparameters
            train_stats: Training statistics (n_samples, n_features, etc.)

        Returns:
            Path to the checkpoint directory
        """
        checkpoint_dir = self.get_model_checkpoint_dir(model_name, create=True)

        # Save model
        model_path = checkpoint_dir / f"{model_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"✓ Saved model to {model_path}")

        # Save metrics
        metrics_path = checkpoint_dir / "metrics.json"
        metrics_serialized = {
            k: (float(v) if isinstance(v, (int, np.integer, np.floating)) else v)
            for k, v in metrics.items()
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_serialized, f, indent=2)
        logger.info(f"✓ Saved metrics to {metrics_path}")

        # Save CV results if available
        if cv_results:
            cv_path = checkpoint_dir / "cv_results.json"
            cv_serialized = {}
            for k, v in cv_results.items():
                if isinstance(v, np.ndarray):
                    cv_serialized[k] = v.tolist()
                elif isinstance(v, (float, np.floating)):
                    cv_serialized[k] = float(v)
                elif isinstance(v, (int, np.integer)):
                    cv_serialized[k] = int(v)
                else:
                    cv_serialized[k] = v
            with open(cv_path, "w") as f:
                json.dump(cv_serialized, f, indent=2)
            logger.info(f"✓ Saved CV results to {cv_path}")

        # Save feature importance if available
        if feature_importance is not None and isinstance(feature_importance, pd.DataFrame):
            fi_path = checkpoint_dir / "feature_importance.csv"
            feature_importance.to_csv(fi_path, index=False)
            logger.info(f"✓ Saved feature importance to {fi_path}")

        # Save hyperparameters
        if hyperparams:
            hp_path = checkpoint_dir / "hyperparameters.json"
            hp_serialized = {}
            for k, v in hyperparams.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    hp_serialized[k] = v
                else:
                    hp_serialized[k] = str(v)
            with open(hp_path, "w") as f:
                json.dump(hp_serialized, f, indent=2)
            logger.info(f"✓ Saved hyperparameters to {hp_path}")

        # Save training statistics
        if train_stats:
            stats_path = checkpoint_dir / "training_stats.json"
            stats_serialized = {}
            for k, v in train_stats.items():
                if isinstance(v, (int, np.integer)):
                    stats_serialized[k] = int(v)
                elif isinstance(v, (float, np.floating)):
                    stats_serialized[k] = float(v)
                else:
                    stats_serialized[k] = v
            with open(stats_path, "w") as f:
                json.dump(stats_serialized, f, indent=2)
            logger.info(f"✓ Saved training stats to {stats_path}")

        # Create summary file
        summary = {
            "model_name": model_name,
            "timestamp": self.timestamp,
            "metrics": metrics_serialized,
            "n_samples": train_stats.get("n_samples") if train_stats else None,
            "n_features": train_stats.get("n_features") if train_stats else None,
        }
        summary_path = checkpoint_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Saved checkpoint summary to {summary_path}")

        logger.info(f"✅ Model checkpoint saved to: {checkpoint_dir}")
        return checkpoint_dir

    def load_model_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load a saved model checkpoint with all metadata.

        Args:
            checkpoint_path: Path to the checkpoint directory

        Returns:
            Dictionary containing model, metrics, cv_results, feature_importance, etc.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

        result = {}

        # Load model
        model_files = list(checkpoint_path.glob("*_model.pkl"))
        if model_files:
            model_path = model_files[0]
            with open(model_path, "rb") as f:
                result["model"] = pickle.load(f)
            logger.info(f"✓ Loaded model from {model_path}")

        # Load metrics
        metrics_path = checkpoint_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                result["metrics"] = json.load(f)
            logger.info(f"✓ Loaded metrics from {metrics_path}")

        # Load CV results
        cv_path = checkpoint_path / "cv_results.json"
        if cv_path.exists():
            with open(cv_path, "r") as f:
                result["cv_results"] = json.load(f)
            logger.info(f"✓ Loaded CV results from {cv_path}")

        # Load feature importance
        fi_path = checkpoint_path / "feature_importance.csv"
        if fi_path.exists():
            result["feature_importance"] = pd.read_csv(fi_path)
            logger.info(f"✓ Loaded feature importance from {fi_path}")

        # Load hyperparameters
        hp_path = checkpoint_path / "hyperparameters.json"
        if hp_path.exists():
            with open(hp_path, "r") as f:
                result["hyperparameters"] = json.load(f)
            logger.info(f"✓ Loaded hyperparameters from {hp_path}")

        # Load training statistics
        stats_path = checkpoint_path / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                result["training_stats"] = json.load(f)
            logger.info(f"✓ Loaded training stats from {stats_path}")

        logger.info(f"✅ Model checkpoint loaded from: {checkpoint_path}")
        return result

    def list_checkpoints(self) -> Dict[str, list]:
        """
        List all available checkpoints organized by model type.

        Returns:
            Dictionary mapping model names to list of checkpoint timestamps
        """
        checkpoints = {}

        for timestamp_dir in sorted(self.base_dir.iterdir()):
            if not timestamp_dir.is_dir():
                continue

            for model_dir in timestamp_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name
                if model_name not in checkpoints:
                    checkpoints[model_name] = []

                checkpoints[model_name].append(timestamp_dir.name)

        return checkpoints

    def get_latest_checkpoint(self, model_name: str) -> Optional[Path]:
        """
        Get the latest checkpoint for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Path to the latest checkpoint or None if not found
        """
        checkpoint_dirs = []

        for timestamp_dir in self.base_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            model_dir = timestamp_dir / model_name
            if model_dir.exists():
                checkpoint_dirs.append(model_dir)

        if not checkpoint_dirs:
            return None

        # Return the most recently created checkpoint
        return max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)

    def create_checkpoint_report(self, output_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Create a report of all checkpoints with their metrics.

        Args:
            output_file: Optional path to save the report as CSV

        Returns:
            DataFrame with checkpoint information
        """
        checkpoints_data = []

        for timestamp_dir in sorted(self.base_dir.iterdir()):
            if not timestamp_dir.is_dir():
                continue

            for model_dir in timestamp_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                summary_path = model_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        summary = json.load(f)

                    metrics_path = model_dir / "metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                    else:
                        metrics = {}

                    checkpoints_data.append({
                        "timestamp": summary.get("timestamp"),
                        "model_name": summary.get("model_name"),
                        "n_samples": summary.get("n_samples"),
                        "n_features": summary.get("n_features"),
                        "avg_f1_macro": metrics.get("avg_f1_macro"),
                        "avg_accuracy": metrics.get("avg_accuracy"),
                        "best_f1_macro": metrics.get("best_f1_macro"),
                        "checkpoint_path": str(model_dir),
                    })

        df_report = pd.DataFrame(checkpoints_data)

        if output_file and not df_report.empty:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df_report.to_csv(output_file, index=False)
            logger.info(f"✓ Checkpoint report saved to {output_file}")

        return df_report
