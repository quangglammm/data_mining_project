"""File-based model repository implementation."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from ...domain.repositories.model_repository import ModelRepository
from ...domain.use_cases.model_checkpoint_manager import ModelCheckpointManager

logger = logging.getLogger(__name__)


class FileModelRepository(ModelRepository):
    """Repository for saving/loading ML models to/from files."""

    def __init__(self, model_dir: str = "models"):
        """
        Initialize repository.

        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model to file."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = self.model_dir / f"model_{timestamp}.pkl"
        metadata_file = self.model_dir / f"model_{timestamp}_metadata.pkl"

        logger.info(f"Saving model to {model_file}")

        # Save model
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Save metadata if provided
        if metadata:
            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f)

        logger.info(f"Model saved successfully: {model_file}")
        return str(model_file)

    def load_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Load model from file.
        
        Args:
            model_id: Path to model file. If None, loads the latest model.
            
        Returns:
            Dictionary with 'model' and 'metadata' keys
        """
        # If no model_id provided, find the latest model file.
        if model_id is None:
            model_files = sorted([f for f in self.model_dir.glob("model_*.pkl") if not f.stem.endswith("_metadata")])
            if model_files:
                model_file = model_files[-1]  # Get the most recent
                logger.info(f"No model specified, using latest: {model_file.name}")
            else:
                # No direct model files saved — attempt to fall back to checkpoints
                try:
                    mgr = ModelCheckpointManager(base_checkpoint_dir=str(self.model_dir / "checkpoints") if (self.model_dir / "checkpoints").exists() else "models/checkpoints")
                    # prefer 'logistic_regression' checkpoint if available
                    preferred_names = ["logistic_regression", "logistic", "lr"]
                    ckpt_dir = None
                    for name in preferred_names:
                        ckpt = mgr.get_latest_checkpoint(name)
                        if ckpt is not None:
                            ckpt_dir = ckpt
                            selected_name = name
                            break

                    if ckpt_dir is not None:
                        logger.info(f"No model file found; loading latest checkpoint for '{selected_name}': {ckpt_dir}")
                        ckpt_data = mgr.load_model_checkpoint(ckpt_dir)
                        model = ckpt_data.get("model")
                        # Compose metadata from checkpoint files
                        metadata = {}
                        if "metrics" in ckpt_data:
                            metadata["metrics"] = ckpt_data["metrics"]
                        if "training_stats" in ckpt_data:
                            metadata.update(ckpt_data.get("training_stats", {}))
                        if "hyperparameters" in ckpt_data:
                            metadata["hyperparameters"] = ckpt_data["hyperparameters"]
                        # Return early since we've loaded from checkpoint
                        logger.info("Model loaded from checkpoint successfully")
                        return {"model": model, "metadata": metadata}
                except Exception as e:
                    logger.warning(f"Checkpoint fallback failed: {e}")
                raise FileNotFoundError(f"No models found in {self.model_dir} and no checkpoints available")
        else:
            model_file = Path(model_id)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_id}")

        logger.info(f"Loading model from {model_file}")

        # Load model
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Load metadata
        metadata_file = model_file.parent / f"{model_file.stem}_metadata.pkl"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
            logger.info(f"Metadata loaded from {metadata_file.name}")
        else:
            logger.warning(f"No metadata file found: {metadata_file}")
            # If the model file appears to be inside a checkpoint directory, try to load checkpoint metadata
            try:
                # model_file.parent is expected to be the checkpoint directory
                ckpt_dir = model_file.parent
                # Basic heuristic: checkpoint directories contain metrics.json or summary.json
                if (ckpt_dir / "metrics.json").exists() or (ckpt_dir / "summary.json").exists():
                    mgr = ModelCheckpointManager()
                    ckpt_data = mgr.load_model_checkpoint(ckpt_dir)
                    # Compose metadata from checkpoint data
                    if "metrics" in ckpt_data:
                        metadata["metrics"] = ckpt_data["metrics"]
                    if "training_stats" in ckpt_data:
                        metadata.update(ckpt_data.get("training_stats", {}))
                    if "hyperparameters" in ckpt_data:
                        metadata["hyperparameters"] = ckpt_data.get("hyperparameters")
                    # Try to extract feature names from feature_importance if available
                    fi = ckpt_data.get("feature_importance")
                    if fi is not None and hasattr(fi, "columns"):
                        if "feature" in fi.columns:
                            metadata["feature_names"] = fi["feature"].tolist()
                        else:
                            metadata["feature_names"] = fi.columns.tolist()
                    logger.info(f"Checkpoint metadata assembled from {ckpt_dir}")
            except Exception as e:
                logger.debug(f"Checkpoint metadata extraction failed: {e}")

        logger.info("Model loaded successfully")
        return {"model": model, "metadata": metadata}

    def model_exists(self, model_id: str) -> bool:
        """Check if model file exists."""
        return Path(model_id).exists()

