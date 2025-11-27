"""Use case for prediction + rich, human-readable explanation using contrast patterns."""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import shap
import shap.maskers

logger = logging.getLogger(__name__)


class PredictAndExplainUseCase:
    """Predict yield class and explain using contrast patterns + SHAP."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_labels: np.ndarray,
        contrast_patterns: Optional[Union[Tuple[Tuple[str, ...], ...], set]] = None,
        contrast_report_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize with trained model and contrast patterns for symbolic explanation.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            class_labels: Array of class labels
            contrast_patterns: Tuple of tuples (ordered) or set (will be converted to ordered tuple)
            contrast_report_df: DataFrame with pattern information
        """
        self.model = model
        self.feature_names = feature_names
        self.class_labels = np.array(class_labels)
        
        # Convert patterns to ordered tuple for determinism
        if contrast_patterns is None:
            self.contrast_patterns: Tuple[Tuple[str, ...], ...] = ()
        elif isinstance(contrast_patterns, tuple):
            self.contrast_patterns = contrast_patterns
        else:
            # Convert set to ordered tuple
            self.contrast_patterns = tuple(sorted(contrast_patterns, key=lambda x: (len(x), x)))
        
        self.contrast_report_df = contrast_report_df

        # Build pattern → column name mapping (maintains order from tuple)
        self.pattern_to_col = {}
        for i, pat in enumerate(self.contrast_patterns):
            col_name = f"pat_{i:03d}__{'__'.join(pat)}"
            self.pattern_to_col[pat] = col_name

        logger.info(
            f"PredictAndExplainUseCase ready: {len(self.pattern_to_col)} contrast patterns loaded (ordered)"
        )

    def _compute_shap_features(
        self, X: pd.DataFrame, pred_idx: int, top_n_features: int = 6
    ) -> Dict[str, float]:
        """SHAP for multiclass LogisticRegression — Manual per-class explainer (bug-proof)"""
        top_features = {}

        try:
            X_numeric = X.select_dtypes(include=[np.number]).copy()
            if X_numeric.empty:
                logger.info("No numeric features for SHAP")
                return top_features

            if not hasattr(self.model, "named_steps") or "lr" not in self.model.named_steps:
                logger.warning("Model is not a LogisticRegression pipeline — skipping SHAP")
                return top_features

            lr = self.model.named_steps["lr"]
            X_transformed = self.model[:-1].transform(X_numeric)  # Scaled features (1, 52)

            # MANUAL PER-CLASS EXPLAINER — Directly from SHAP source (fixes multiclass list bug)
            # Extract coef/intercept for the predicted class only
            if hasattr(lr, 'classes_') and pred_idx < len(lr.classes_):
                class_coef = lr.coef_[pred_idx]  # Shape: (52,)
                class_intercept = lr.intercept_[pred_idx]  # Scalar
            else:
                # Fallback for binary/single-class
                class_coef = lr.coef_[0] if len(lr.coef_.shape) > 1 else lr.coef_
                class_intercept = lr.intercept_[0] if hasattr(lr.intercept_, '__len__') else lr.intercept_

            # Build single-class linear model (coef, intercept) — avoids multiclass list
            single_class_model = (class_coef, class_intercept)

            # Use Independent masker on transformed data
            masker = shap.maskers.Independent(X_transformed)

            # Create explainer for THIS CLASS ONLY — no list, no concatenation
            explainer = shap.LinearExplainer(
                single_class_model,
                masker,
                link=shap.links.identity  # For probability scale in multiclass
            )

            # Compute SHAP — now returns SINGLE array (1, 52), not list
            shap_values = explainer.shap_values(X_transformed)

            # Flatten to 1D (52 values)
            sv = np.array(shap_values).flatten()

            # Verify length (should be exactly 52)
            if len(sv) != len(X_numeric.columns):
                logger.warning(f"SHAP length mismatch: {len(sv)} vs {len(X_numeric.columns)}")
                return top_features

            # Rank and format top features
            importance = pd.Series(np.abs(sv), index=X_numeric.columns)
            top_idx = importance.nlargest(top_n_features).index

            for feat in top_idx:
                val = float(sv[X_numeric.columns.get_loc(feat)])
                if feat.startswith("pat_"):
                    readable = feat.split("__", 1)[1].replace("__", " → ")
                    top_features[f"Weather Pattern: {readable}"] = round(val, 8)
                else:
                    top_features[feat.replace("_", " ").title()] = round(val, 8)

            logger.debug(f"SHAP succeeded (manual per-class): {len(top_features)} features for class {self.class_labels[pred_idx]}")
            return top_features

        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            return top_features

    def _get_triggered_patterns(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Find which contrast patterns are active in this season."""
        triggered = []
        for pattern, col_name in self.pattern_to_col.items():
            if col_name in row.index and row[col_name] == 1:
                # Look up growth rate and type from report
                if self.contrast_report_df is not None:
                    match = self.contrast_report_df[
                        self.contrast_report_df["events"].apply(lambda x: tuple(x) == pattern)
                    ]
                    if not match.empty:
                        r = match.iloc[0]
                        triggered.append(
                            {
                                "pattern": " → ".join(pattern),
                                "growth_rate": round(r["growth_rate"], 2),
                                "type": r["type"],
                                "strength": r.get("strength", "moderate"),
                            }
                        )
                else:
                    triggered.append(
                        {
                            "pattern": " → ".join(pattern),
                            "growth_rate": None,
                            "type": "unknown",
                            "strength": "unknown",
                        }
                    )
        return triggered

    def execute(
        self,
        X: pd.DataFrame,
        top_n_features: int = 6,
        use_shap: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict and generate rich explanation.
        """
        logger.info(f"Predicting for {len(X)} season(s)")

        # Clean input
        X_clean = X.copy()
        if "year" in X_clean.columns:
            X_clean = X_clean.drop(columns=["year"])

        # Predict
        y_pred = self.model.predict(X_clean)
        y_pred_proba = (
            self.model.predict_proba(X_clean) if hasattr(self.model, "predict_proba") else None
        )

        # Cast predicted label to category early
        pred_class = str(self.class_labels[y_pred[0]])
        proba = y_pred_proba[0].tolist() if y_pred_proba is not None else None
        max_confidence = max(proba) if proba else None

        # Determine confidence level
        confidence_level = "Unknown"
        if max_confidence:
            if max_confidence >= 0.70:
                confidence_level = "High"
            elif max_confidence >= 0.50:
                confidence_level = "Medium"
            else:
                confidence_level = "Low"

        # Add warnings for low confidence
        warnings = []
        if max_confidence and max_confidence < 0.50:
            warnings.append(
                "⚠️ Low confidence prediction. Model is uncertain - "
                "consider waiting for more complete weather data or "
                "obtaining field observations."
            )

            # Check if probabilities are roughly equal
            proba_sorted = sorted(proba, reverse=True)
            if proba_sorted[0] - proba_sorted[1] < 0.15:  # Top 2 classes within 15%
                top2_classes = [
                    self.class_labels[i]
                    for i in sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)[:2]
                ]
                warnings.append(
                    f"Model is torn between {top2_classes[0]} and {top2_classes[1]} "
                    f"({proba_sorted[0]:.1%} vs {proba_sorted[1]:.1%})"
                )

        result = {
            "prediction": pred_class,
            "confidence": round(max(proba), 3) if proba else None,
            "confidence_level": confidence_level,
            "warnings": warnings,
            "probabilities": (
                {self.class_labels[i]: round(p, 3) for i, p in enumerate(proba)} if proba else None
            ),
        }

        # === 1. Symbolic Explanation: Triggered Contrast Patterns ===
        row = X_clean.iloc[0]
        triggered = self._get_triggered_patterns(row)

        high_yield_patterns = [t for t in triggered if "High" in t["type"]]
        low_yield_patterns = [t for t in triggered if "Low" in t["type"]]

        explanation_lines = []

        if pred_class == "High" and high_yield_patterns:
            top_pat = sorted(
                high_yield_patterns, key=lambda x: x["growth_rate"] or 0, reverse=True
            )[0]
            explanation_lines.append(
                f"This season matches {len(high_yield_patterns)} high-yield weather pattern(s)"
            )
            explanation_lines.append(
                f"Strongest: {top_pat['pattern']} "
                f"(typically {top_pat['growth_rate']}× more in High yield)"
            )
        elif pred_class == "Low" and low_yield_patterns:
            top_pat = sorted(low_yield_patterns, key=lambda x: x["growth_rate"] or 0, reverse=True)[
                0
            ]
            explanation_lines.append(
                f"This season shows {len(low_yield_patterns)} risk pattern(s) linked to Low yield"
            )
            explanation_lines.append(
                f"Strongest risk: {top_pat['pattern']} "
                f"(typically {top_pat['growth_rate']}× more in Low yield)"
            )
        else:
            explanation_lines.append("No strong symbolic weather patterns detected.")

        result["explanation"] = " | ".join(explanation_lines)
        result["triggered_patterns"] = triggered

        # === 2. SHAP Explanation (numerical + pattern features) ===
        # Always compute SHAP for every prediction (regardless of confidence level)
        top_features = {}
        if use_shap:
            top_features = self._compute_shap_features(
                X_clean.iloc[0:1], y_pred[0], top_n_features
            )
        
        result["top_features"] = top_features if top_features else None

        logger.info(f"Prediction: {pred_class} | Explanation ready")
        return result