"""Use case for prediction + rich, human-readable explanation using contrast patterns."""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
import shap

logger = logging.getLogger(__name__)


class PredictAndExplainUseCase:
    """Predict yield class and explain using contrast patterns + SHAP."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        class_labels: np.ndarray,
        contrast_patterns: Optional[Set[Tuple[str, ...]]] = None,
        contrast_report_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize with trained model and contrast patterns for symbolic explanation.
        """
        self.model = model
        self.feature_names = feature_names
        self.class_labels = np.array(class_labels)
        self.contrast_patterns = contrast_patterns or set()
        self.contrast_report_df = contrast_report_df

        # Build pattern → column name mapping
        self.pattern_to_col = {}
        if contrast_patterns:
            for i, pat in enumerate(sorted(contrast_patterns, key=lambda x: (len(x), x))):
                col_name = f"pat_{i:03d}__{'__'.join(pat)}"
                self.pattern_to_col[pat] = col_name

        logger.info(
            f"PredictAndExplainUseCase ready: {len(self.pattern_to_col)} contrast patterns loaded"
        )

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

        pred_class = self.class_labels[y_pred[0]]
        proba = y_pred_proba[0].tolist() if y_pred_proba is not None else None

        result = {
            "prediction": pred_class,
            "confidence": round(max(proba), 3) if proba else None,
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
        top_features = {}
        if use_shap and hasattr(self.model, "predict_proba"):
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_clean.iloc[0:1])

                if isinstance(shap_values, list):
                    shap_vals = shap_values[y_pred[0]][0]
                else:
                    shap_vals = shap_values[0]

                importance = pd.Series(np.abs(shap_vals), index=X_clean.columns)
                top_idx = importance.nlargest(top_n_features).index

                for feat in top_idx:
                    val = shap_vals[X_clean.columns.get_loc(feat)]
                    # Make pattern names readable
                    if feat.startswith("pat_"):
                        readable = " → ".join(feat.split("__")[1:])
                        top_features[f"Weather Pattern: {readable}"] = round(val, 4)
                    else:
                        top_features[feat.replace("_", " ").title()] = round(val, 4)

                result["top_features"] = top_features

            except Exception as e:
                logger.warning(f"SHAP failed: {e}")
                result["top_features"] = None
        else:
            result["top_features"] = None

        logger.info(f"Prediction: {pred_class} | Explanation ready")
        return result
