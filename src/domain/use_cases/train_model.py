"""Use case for training ML model."""

import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

logger = logging.getLogger(__name__)


class TrainModelUseCase:
    """Use case to train ML model with time series cross-validation."""

    def __init__(
        self,
        n_splits: int = 5,
        model_type: str = "xgboost",
        random_state: int = 42,
    ):
        """
        Initialize use case.

        Args:
            n_splits: Number of time series splits
            model_type: Type of model ('xgboost' or 'random_forest')
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.model_type = model_type
        self.random_state = random_state

    def execute(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        class_labels: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute model training.

        Args:
            X: Feature matrix
            y: Target labels (encoded)
            class_labels: Original class labels

        Returns:
            Tuple of (trained_model, evaluation_metrics)
        """
        logger.info(
            f"Training {self.model_type} model with {self.n_splits} time series splits"
        )

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        f1_scores = []
        acc_scores = []

        # Train on each fold
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold + 1}/{self.n_splits}")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            train_years = X_train["year"] if "year" in X_train.columns else None
            test_years = X_test["year"] if "year" in X_test.columns else None

            if train_years is not None:
                logger.info(
                    f"Training: {len(X_train)} samples "
                    f"(Years ~{train_years.min()}-{train_years.max()})"
                )
                logger.info(
                    f"Testing: {len(X_test)} samples "
                    f"(Years ~{test_years.min()}-{test_years.max()})"
                )

            # Remove year column if present
            X_train_clean = X_train.drop(columns=["year"]) if "year" in X_train.columns else X_train
            X_test_clean = X_test.drop(columns=["year"]) if "year" in X_test.columns else X_test

            # Train model
            if self.model_type == "xgboost":
                model = xgb.XGBClassifier(
                    objective="multi:softmax",
                    num_class=len(class_labels),
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    random_state=self.random_state,
                )
            else:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                )

            model.fit(X_train_clean, y_train)

            # Evaluate
            y_pred = model.predict(X_test_clean)
            f1 = f1_score(y_test, y_pred, average="macro")
            acc = accuracy_score(y_test, y_pred)
            f1_scores.append(f1)
            acc_scores.append(acc)

            logger.info(f"Fold {fold + 1} F1-Score (macro): {f1:.4f}")
            logger.info(f"Fold {fold + 1} Accuracy: {acc:.4f}")

            # Store last fold model and predictions
            if fold == self.n_splits - 1:
                last_model = model
                last_y_test = y_test
                last_y_pred = y_pred

        # Train final model on full data
        X_clean = X.drop(columns=["year"]) if "year" in X.columns else X
        if self.model_type == "xgboost":
            final_model = xgb.XGBClassifier(
                objective="multi:softmax",
                num_class=len(class_labels),
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=self.random_state,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier

            final_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
            )

        final_model.fit(X_clean, y)

        # Compile metrics
        metrics = {
            "avg_f1_macro": np.mean(f1_scores),
            "std_f1_macro": np.std(f1_scores),
            "avg_accuracy": np.mean(acc_scores),
            "std_accuracy": np.std(acc_scores),
            "f1_scores": f1_scores,
            "acc_scores": acc_scores,
            "last_fold_report": classification_report(
                last_y_test, last_y_pred, target_names=class_labels, output_dict=True
            ),
        }

        logger.info(
            f"Average F1-Score (macro): {metrics['avg_f1_macro']:.4f} "
            f"+/- {metrics['std_f1_macro']:.4f}"
        )
        logger.info(
            f"Average Accuracy: {metrics['avg_accuracy']:.4f} "
            f"+/- {metrics['std_accuracy']:.4f}"
        )

        return final_model, metrics

