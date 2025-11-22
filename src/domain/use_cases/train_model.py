"""TrainModelUseCase — XGBoost 3.0+ compatible with native early stopping."""

import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class TrainModelUseCase:
    """Train high-performance XGBoost with TimeSeries CV + native early stopping (XGBoost 3.0+)."""

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.random_state = random_state

        self.params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "gamma": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    def execute(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        class_labels: np.ndarray,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        logger.info("Starting optimized XGBoost training (XGBoost 3.0+ native early stopping)")

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        f1_scores = []
        acc_scores = []
        best_f1 = 0.0
        best_model = None

        X_clean = X.drop(columns=["year"], errors="ignore")
        has_year = "year" in X.columns
        years = X["year"] if has_year else None

        # CV: Use native xgb.train() for full early stopping
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            logger.info(f"Training Fold {fold + 1}/{self.n_splits}")

            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if has_year:
                logger.info(f"  Train: {years.iloc[train_idx].min()}–{years.iloc[train_idx].max()}")
                logger.info(f"  Valid: {years.iloc[val_idx].min()}–{years.iloc[val_idx].max()}")

            # Native early stopping with DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            fold_model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=1000,
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            # Predict with native model
            y_pred = (fold_model.predict(dval) * 1000).argmax(axis=1)  # softprob → class
            f1 = f1_score(y_val, y_pred, average="macro")
            acc = accuracy_score(y_val, y_pred)

            f1_scores.append(f1)
            acc_scores.append(acc)
            logger.info(
                f"  Fold {fold + 1} → F1-macro: {f1:.4f} | Acc: {acc:.4f} | Trees: {fold_model.num_boosted_rounds()}"
            )

            if f1 > best_f1:
                best_f1 = f1
                # Save best booster for final model
                best_booster = fold_model

        # Final model: Train on full data with best params
        logger.info("Training final model on full dataset...")
        dtrain_full = xgb.DMatrix(X_clean, label=y)
        final_booster = xgb.train(
            params=self.params,
            dtrain=dtrain_full,
            num_boost_round=1000,
            verbose_eval=False,
        )

        # Convert best booster to XGBClassifier (for scikit-learn compatibility)
        if best_booster:
            final_model = xgb.XGBClassifier(**self.params)
            final_model._Booster = best_booster
            final_model.fit(X_clean, y)  # Dummy fit to set attributes
        else:
            final_model = xgb.XGBClassifier(**self.params)
            final_model.fit(X_clean, y)

        metrics = {
            "avg_f1_macro": np.mean(f1_scores),
            "std_f1_macro": np.std(f1_scores),
            "best_f1_macro": best_f1,
            "avg_accuracy": np.mean(acc_scores),
            "n_features": X_clean.shape[1],
            "n_samples": len(X_clean),
            "best_n_trees": final_booster.num_boosted_rounds(),
        }

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETE — HIGH-PERFORMANCE MODEL READY")
        logger.info(f"   Best F1-macro:     {best_f1:.4f}")
        logger.info(
            f"   Avg F1-macro:      {metrics['avg_f1_macro']:.4f} ± {metrics['std_f1_macro']:.3f}"
        )
        logger.info(f"   Avg Accuracy:      {metrics['avg_accuracy']:.4f}")
        logger.info(f"   Features used:     {X_clean.shape[1]} (numeric + contrast patterns)")
        logger.info(f"   Final trees:       {metrics['best_n_trees']}")
        logger.info("=" * 70)

        return final_model, metrics
