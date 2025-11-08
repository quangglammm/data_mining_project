"""Model repository interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pickle


class ModelRepository(ABC):
    """Abstract repository for ML model persistence."""

    @abstractmethod
    def save_model(self, model: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a trained model.

        Args:
            model: The model object to save
            metadata: Optional metadata about the model

        Returns:
            Path or identifier where model was saved
        """
        pass

    @abstractmethod
    def load_model(self, model_id: str) -> Any:
        """
        Load a trained model.

        Args:
            model_id: Model identifier or path

        Returns:
            The loaded model object
        """
        pass

    @abstractmethod
    def model_exists(self, model_id: str) -> bool:
        """
        Check if a model exists.

        Args:
            model_id: Model identifier or path

        Returns:
            True if model exists, False otherwise
        """
        pass

