"""File-based model repository implementation."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from ...domain.repositories.model_repository import ModelRepository

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
        # If no model_id provided, find the latest model
        if model_id is None:
            model_files = sorted([f for f in self.model_dir.glob("model_*.pkl") if not f.stem.endswith("_metadata")])
            if not model_files:
                raise FileNotFoundError(f"No models found in {self.model_dir}")
            model_file = model_files[-1]  # Get the most recent
            logger.info(f"No model specified, using latest: {model_file.name}")
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

        logger.info("Model loaded successfully")
        return {"model": model, "metadata": metadata}

    def model_exists(self, model_id: str) -> bool:
        """Check if model file exists."""
        return Path(model_id).exists()

