"""Use case for prediction and explanation."""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import shap

logger = logging.getLogger(__name__)


class PredictAndExplainUseCase:
    """Use case to predict yield class and generate explanations."""

    def __init__(self, model: Any, feature_names: List[str], class_labels: np.ndarray):
        """
        Initialize use case.

        Args:
            model: Trained ML model
            feature_names: List of feature names
            class_labels: Class label names
        """
        self.model = model
        self.feature_names = feature_names
        self.class_labels = class_labels

    def execute(
        self,
        X: pd.DataFrame,
        top_n_features: int = 5,
        use_shap: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute prediction and explanation.

        Args:
            X: Feature matrix for prediction
            top_n_features: Number of top features to return
            use_shap: Whether to use SHAP for explanation

        Returns:
            Dictionary with prediction and explanation
        """
        logger.info(f"Predicting for {len(X)} samples")

        # Remove year column if present
        X_clean = X.drop(columns=["year"]) if "year" in X.columns else X

        # Predict
        y_pred = self.model.predict(X_clean)
        y_pred_proba = (
            self.model.predict_proba(X_clean)
            if hasattr(self.model, "predict_proba")
            else None
        )

        # Get predicted class name
        predicted_class = self.class_labels[y_pred[0]] if len(y_pred) == 1 else [
            self.class_labels[p] for p in y_pred
        ]

        result = {
            "prediction": predicted_class,
            "prediction_encoded": y_pred[0] if len(y_pred) == 1 else y_pred.tolist(),
            "probabilities": (
                y_pred_proba[0].tolist() if y_pred_proba is not None else None
            ),
        }

        # Generate SHAP explanation if requested
        if use_shap and hasattr(self.model, "predict_proba"):
            try:
                logger.info("Generating SHAP explanation")
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_clean.iloc[0:1])

                # Handle multi-class SHAP values
                if isinstance(shap_values, list):
                    # Multi-class: get values for predicted class
                    class_idx = y_pred[0]
                    shap_vals = shap_values[class_idx][0]
                else:
                    shap_vals = shap_values[0]

                # Get top features
                feature_importance = pd.Series(
                    np.abs(shap_vals), index=self.feature_names
                ).sort_values(ascending=False)

                top_features = feature_importance.head(top_n_features).to_dict()
                result["top_features"] = top_features
                result["shap_values"] = shap_vals.tolist()

                logger.info(f"Top {top_n_features} features: {list(top_features.keys())}")

            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                result["top_features"] = None
                result["shap_values"] = None
        else:
            # Fallback to feature importance
            if hasattr(self.model, "feature_importances_"):
                feature_importance = pd.Series(
                    self.model.feature_importances_, index=self.feature_names
                ).sort_values(ascending=False)
                top_features = feature_importance.head(top_n_features).to_dict()
                result["top_features"] = top_features
            else:
                result["top_features"] = None

        return result

