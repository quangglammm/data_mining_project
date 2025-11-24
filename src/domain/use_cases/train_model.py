"""TrainModelUseCase — XGBoost 3.0+ compatible with native early stopping."""

import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import StratifiedKFold

from src.domain.use_cases.optimize_hyperparameters import tune_hyperparameters

# Add these imports at the top of your file, after the existing imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import os


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
            "learning_rate": 0.01,
            "max_depth": 3,
            "min_child_weight": 15,
            "gamma": 1.0,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }

    def execute(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        class_labels: np.ndarray,
        tune_hyperparams: bool = False,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        logger.info("Starting optimized XGBoost training (XGBoost 3.0+ native early stopping)")

        # CRITICAL: Verify X and y alignment FIRST
        if len(X) != len(y):
            raise ValueError(
                f"Feature matrix and target misaligned! "
                f"X has {len(X)} rows but y has {len(y)} labels. "
                f"This indicates a bug in BuildFeatureMatrixUseCase."
            )

        logger.info(f"✓ Verified: X and y both have {len(X)} samples")

        # Check for any NaN or infinite values
        if X.isnull().any().any():
            n_nan = X.isnull().sum().sum()
            logger.warning(f"Found {n_nan} NaN values in features - filling with 0")
            X = X.fillna(0)

        if np.isinf(X.select_dtypes(include=[np.number]).values).any():
            logger.warning("Found infinite values in features - clipping")
            X = X.replace([np.inf, -np.inf], 0)

        # Check overall class distribution
        class_counts = Counter(y)
        logger.info(f"Overall class distribution: {dict(class_counts)}")

        min_class_count = min(class_counts.values())

        # Adjust n_splits based on smallest class
        safe_n_splits = min(self.n_splits, min_class_count // 2)

        if safe_n_splits < self.n_splits:
            logger.warning(
                f"Reducing CV splits from {self.n_splits} to {safe_n_splits} "
                f"due to small class sizes (min={min_class_count})"
            )
            self.n_splits = safe_n_splits

        if self.n_splits < 3:
            logger.error(
                f"Insufficient data for reliable CV! Only {min_class_count} samples "
                f"in smallest class. Consider:"
            )
            logger.error("  1. Merge Low+Medium into single 'Below-High' class")
            logger.error("  2. Collect more data")
            logger.error("  3. Use stratified holdout instead of CV")
            raise ValueError("Insufficient samples for cross-validation")

        # tscv = TimeSeriesSplit(n_splits=self.n_splits)
        f1_scores = []
        acc_scores = []
        f1_gaps = []
        best_f1 = 0.0
        best_booster = None

        # Safe handling of 'year' column
        has_year = "year" in X.columns

        if has_year:
            years = X["year"].copy()  # Save BEFORE dropping
            X_clean = X.drop(columns=["year"])
            logger.info(f"Dropped 'year' column: X shape {X.shape} → {X_clean.shape}")
        else:
            X_clean = X.copy()
            years = None

        # CRITICAL: Verify no rows were lost
        assert len(X_clean) == len(
            X
        ), f"Rows lost after dropping column! Before={len(X)}, After={len(X_clean)}"

        assert len(X_clean) == len(y), f"X and y misaligned! X_clean={len(X_clean)}, y={len(y)}"

        logger.info(f"Training on {len(X_clean)} samples with {X_clean.shape[1]} features")

        if tune_hyperparams:
            logger.info("Running hyperparameter tuning...")
            best_params = tune_hyperparameters(X_clean, y, n_splits=3, n_iter=20)
            self.params.update(best_params)
            logger.info(f"Using tuned parameters: {best_params}")

        # Use Stratified instead of TimeSeries
        logger.warning("Using StratifiedKFold due to temporal class imbalance")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)


        # Test if numeric features alone work
        logger.info("Testing baseline models...")

        # Random baseline
        dummy = DummyClassifier(strategy='stratified')
        dummy.fit(X_clean, y)
        baseline_score = dummy.score(X_clean, y)
        logger.info(f"Random baseline: {baseline_score:.4f}")

        # Numeric features only (no patterns)
        numeric_cols = [c for c in X_clean.columns if not c.startswith('pat_')]
        if numeric_cols:
            rf_numeric = RandomForestClassifier(max_depth=3, n_estimators=50, random_state=42)
            rf_numeric.fit(X_clean[numeric_cols], y)
            numeric_score = rf_numeric.score(X_clean[numeric_cols], y)
            logger.info(f"Numeric-only baseline: {numeric_score:.4f}")

        rf_full = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
        rf_full.fit(X_clean, y)
        full_score = rf_full.score(X_clean, y)
        logger.info(f"Full-feature baseline without CV: {full_score:.4f}")

        # Insert this code block after the line: logger.info(f"Full-feature baseline without CV: {full_score:.4f}")

        # New: Numeric-only CV baseline (RF, to compare fairly with full features)
        logger.info("Computing numeric-only CV baseline...")
        numeric_cv_f1 = []
        numeric_cv_acc = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean[numeric_cols], y)):
            X_train_num = X_clean.iloc[train_idx][numeric_cols]
            X_val_num = X_clean.iloc[val_idx][numeric_cols]
            y_train, y_val = y[train_idx], y[val_idx]
            
            rf_num_fold = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
            rf_num_fold.fit(X_train_num, y_train)
            
            y_val_pred = rf_num_fold.predict(X_val_num)
            numeric_cv_f1.append(f1_score(y_val, y_val_pred, average="macro"))
            numeric_cv_acc.append(accuracy_score(y_val, y_val_pred))

        avg_num_cv_f1 = np.mean(numeric_cv_f1)
        std_num_cv_f1 = np.std(numeric_cv_f1)
        avg_num_cv_acc = np.mean(numeric_cv_acc)
        std_num_cv_acc = np.std(numeric_cv_acc)
        logger.info(f"Numeric-only CV baseline - Avg F1-macro: {avg_num_cv_f1:.4f} ± {std_num_cv_f1:.4f}")
        logger.info(f"Numeric-only CV baseline - Avg Accuracy: {avg_num_cv_acc:.4f} ± {std_num_cv_acc:.4f}")

        # New: Dummy Classifier CV baseline (stratified, for fair CV comparison)
        logger.info("Computing Dummy Classifier CV baseline...")
        dummy_cv_f1 = []
        dummy_cv_acc = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            dummy_fold = DummyClassifier(strategy='stratified', random_state=42)
            dummy_fold.fit(X_train, y_train)
            
            y_val_pred = dummy_fold.predict(X_val)
            dummy_cv_f1.append(f1_score(y_val, y_val_pred, average="macro"))
            dummy_cv_acc.append(accuracy_score(y_val, y_val_pred))

        avg_dummy_cv_f1 = np.mean(dummy_cv_f1)
        std_dummy_cv_f1 = np.std(dummy_cv_f1)
        avg_dummy_cv_acc = np.mean(dummy_cv_acc)
        std_dummy_cv_acc = np.std(dummy_cv_acc)
        logger.info(f"Dummy CV baseline - Avg F1-macro: {avg_dummy_cv_f1:.4f} ± {std_dummy_cv_f1:.4f}")
        logger.info(f"Dummy CV baseline - Avg Accuracy: {avg_dummy_cv_acc:.4f} ± {std_dummy_cv_acc:.4f}")

        # New: Logistic Regression CV baseline
        logger.info("Computing Logistic Regression CV baseline...")
        lr_cv_f1 = []
        lr_cv_acc = []
        lr_cv_gaps = []  # To log overfit gaps
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            lr_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, C=1.0, random_state=42))
            ])
            lr_pipeline.fit(X_train, y_train)
            
            y_train_pred = lr_pipeline.predict(X_train)
            y_val_pred = lr_pipeline.predict(X_val)
            
            train_f1 = f1_score(y_train, y_train_pred, average="macro")
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            f1_gap = train_f1 - val_f1
            
            lr_cv_f1.append(val_f1)
            lr_cv_acc.append(accuracy_score(y_val, y_val_pred))
            lr_cv_gaps.append(f1_gap)

        avg_lr_cv_f1 = np.mean(lr_cv_f1)
        std_lr_cv_f1 = np.std(lr_cv_f1)
        avg_lr_cv_acc = np.mean(lr_cv_acc)
        std_lr_cv_acc = np.std(lr_cv_acc)
        avg_lr_gap = np.mean(lr_cv_gaps)
        logger.info(f"Logistic Regression CV - Avg F1-macro: {avg_lr_cv_f1:.4f} ± {std_lr_cv_f1:.4f}")
        logger.info(f"Logistic Regression CV - Avg Accuracy: {avg_lr_cv_acc:.4f} ± {std_lr_cv_acc:.4f}")
        logger.info(f"Logistic Regression CV - Avg F1 Gap: {avg_lr_gap:.1%}")

        # New: KNN CV baseline
        logger.info("Computing KNN CV baseline...")
        knn_cv_f1 = []
        knn_cv_acc = []
        knn_cv_gaps = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            knn_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
            ])
            knn_pipeline.fit(X_train, y_train)
            
            y_train_pred = knn_pipeline.predict(X_train)
            y_val_pred = knn_pipeline.predict(X_val)
            
            train_f1 = f1_score(y_train, y_train_pred, average="macro")
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            f1_gap = train_f1 - val_f1
            
            knn_cv_f1.append(val_f1)
            knn_cv_acc.append(accuracy_score(y_val, y_val_pred))
            knn_cv_gaps.append(f1_gap)

        avg_knn_cv_f1 = np.mean(knn_cv_f1)
        std_knn_cv_f1 = np.std(knn_cv_f1)
        avg_knn_cv_acc = np.mean(knn_cv_acc)
        std_knn_cv_acc = np.std(knn_cv_acc)
        avg_knn_gap = np.mean(knn_cv_gaps)
        logger.info(f"KNN CV - Avg F1-macro: {avg_knn_cv_f1:.4f} ± {std_knn_cv_f1:.4f}")
        logger.info(f"KNN CV - Avg Accuracy: {avg_knn_cv_acc:.4f} ± {std_knn_cv_acc:.4f}")
        logger.info(f"KNN CV - Avg F1 Gap: {avg_knn_gap:.1%}")

        # New: SVM CV baseline
        logger.info("Computing SVM CV baseline...")
        svm_cv_f1 = []
        svm_cv_acc = []
        svm_cv_gaps = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            svm_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
            ])
            svm_pipeline.fit(X_train, y_train)
            
            y_train_pred = svm_pipeline.predict(X_train)
            y_val_pred = svm_pipeline.predict(X_val)
            
            train_f1 = f1_score(y_train, y_train_pred, average="macro")
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            f1_gap = train_f1 - val_f1
            
            svm_cv_f1.append(val_f1)
            svm_cv_acc.append(accuracy_score(y_val, y_val_pred))
            svm_cv_gaps.append(f1_gap)

        avg_svm_cv_f1 = np.mean(svm_cv_f1)
        std_svm_cv_f1 = np.std(svm_cv_f1)
        avg_svm_cv_acc = np.mean(svm_cv_acc)
        std_svm_cv_acc = np.std(svm_cv_acc)
        avg_svm_gap = np.mean(svm_cv_gaps)
        logger.info(f"SVM CV - Avg F1-macro: {avg_svm_cv_f1:.4f} ± {std_svm_cv_f1:.4f}")
        logger.info(f"SVM CV - Avg Accuracy: {avg_svm_cv_acc:.4f} ± {std_svm_cv_acc:.4f}")
        logger.info(f"SVM CV - Avg F1 Gap: {avg_svm_gap:.1%}")

        # New: LightGBM CV baseline
        logger.info("Computing LightGBM CV baseline...")
        lgb_cv_f1 = []
        lgb_cv_acc = []
        lgb_cv_gaps = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            lgb_fold = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=200, random_state=42, verbose=-1)
            lgb_fold.fit(X_train, y_train)
            
            y_train_pred = lgb_fold.predict(X_train)
            y_val_pred = lgb_fold.predict(X_val)
            
            train_f1 = f1_score(y_train, y_train_pred, average="macro")
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            f1_gap = train_f1 - val_f1
            
            lgb_cv_f1.append(val_f1)
            lgb_cv_acc.append(accuracy_score(y_val, y_val_pred))
            lgb_cv_gaps.append(f1_gap)

        avg_lgb_cv_f1 = np.mean(lgb_cv_f1)
        std_lgb_cv_f1 = np.std(lgb_cv_f1)
        avg_lgb_cv_acc = np.mean(lgb_cv_acc)
        std_lgb_cv_acc = np.std(lgb_cv_acc)
        avg_lgb_gap = np.mean(lgb_cv_gaps)
        logger.info(f"LightGBM CV - Avg F1-macro: {avg_lgb_cv_f1:.4f} ± {std_lgb_cv_f1:.4f}")
        logger.info(f"LightGBM CV - Avg Accuracy: {avg_lgb_cv_acc:.4f} ± {std_lgb_cv_acc:.4f}")
        logger.info(f"LightGBM CV - Avg F1 Gap: {avg_lgb_gap:.1%}")

        # New: MLP (Neural Net) CV baseline
        logger.info("Computing MLP CV baseline...")
        mlp_cv_f1 = []
        mlp_cv_acc = []
        mlp_cv_gaps = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            mlp_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=0.01, random_state=42))
            ])
            mlp_pipeline.fit(X_train, y_train)
            
            y_train_pred = mlp_pipeline.predict(X_train)
            y_val_pred = mlp_pipeline.predict(X_val)
            
            train_f1 = f1_score(y_train, y_train_pred, average="macro")
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            f1_gap = train_f1 - val_f1
            
            mlp_cv_f1.append(val_f1)
            mlp_cv_acc.append(accuracy_score(y_val, y_val_pred))
            mlp_cv_gaps.append(f1_gap)

        avg_mlp_cv_f1 = np.mean(mlp_cv_f1)
        std_mlp_cv_f1 = np.std(mlp_cv_f1)
        avg_mlp_cv_acc = np.mean(mlp_cv_acc)
        std_mlp_cv_acc = np.std(mlp_cv_acc)
        avg_mlp_gap = np.mean(mlp_cv_gaps)
        logger.info(f"MLP CV - Avg F1-macro: {avg_mlp_cv_f1:.4f} ± {std_mlp_cv_f1:.4f}")
        logger.info(f"MLP CV - Avg Accuracy: {avg_mlp_cv_acc:.4f} ± {std_mlp_cv_acc:.4f}")
        logger.info(f"MLP CV - Avg F1 Gap: {avg_mlp_gap:.1%}")

        rf_cv_f1 = []
        rf_cv_acc = []

        # CV: Use native xgb.train() for full early stopping
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_clean, y)):
            logger.info(f"Training Fold {fold + 1}/{self.n_splits}")

            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Check class distribution in this fold
            train_counts = Counter(y_train)
            val_counts = Counter(y_val)

            logger.info(f"  Fold {fold + 1} class distribution:")
            logger.info(f"    Train: {dict(train_counts)}")
            logger.info(f"    Val:   {dict(val_counts)}")

            # Validate all classes present
            missing_in_train = set(range(len(class_labels))) - set(train_counts.keys())
            missing_in_val = set(range(len(class_labels))) - set(val_counts.keys())

            if missing_in_train:
                logger.error(
                    f"    ✗ Classes {[class_labels[i] for i in missing_in_train]} "
                    f"missing from training set!"
                )
                raise ValueError(f"Fold {fold+1} missing classes in training")

            if missing_in_val:
                logger.warning(
                    f"    ⚠️ Classes {[class_labels[i] for i in missing_in_val]} "
                    f"missing from validation set"
                )

            # Check for severe imbalance
            train_ratios = [train_counts[i] / len(y_train) for i in range(len(class_labels))]
            if max(train_ratios) / min(train_ratios) > 5:
                logger.warning(
                    f"    ⚠️ Severe imbalance in fold {fold+1}: "
                    f"ratios = {[f'{r:.1%}' for r in train_ratios]}"
                )

            if has_year:
                logger.info(
                    f"  Train years: {years.iloc[train_idx].min()}–{years.iloc[train_idx].max()}"
                )
                logger.info(
                    f"  Valid years: {years.iloc[val_idx].min()}–{years.iloc[val_idx].max()}"
                )

            rf_fold = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
            rf_fold.fit(X_train, y_train)

            y_val_rf_pred = rf_fold.predict(X_val)
            rf_f1_fold = f1_score(y_val, y_val_rf_pred, average="macro")
            rf_acc_fold = accuracy_score(y_val, y_val_rf_pred)

            rf_cv_f1.append(rf_f1_fold)
            rf_cv_acc.append(rf_acc_fold)

            # Native early stopping with DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            fold_model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=500,
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )

            # 🔧 BUG FIX: Predict on CORRECT datasets
            y_train_pred = fold_model.predict(dtrain).argmax(axis=1)  # ← Fixed: dtrain not dval
            y_val_pred = fold_model.predict(dval).argmax(axis=1)

            # Verify prediction shapes
            assert len(y_train_pred) == len(
                y_train
            ), f"Train prediction mismatch: pred={len(y_train_pred)}, actual={len(y_train)}"
            assert len(y_val_pred) == len(
                y_val
            ), f"Val prediction mismatch: pred={len(y_val_pred)}, actual={len(y_val)}"

            # Calculate metrics for both
            train_f1 = f1_score(y_train, y_train_pred, average="macro")
            val_f1 = f1_score(y_val, y_val_pred, average="macro")
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)

            # Calculate overfitting gap
            f1_gap = train_f1 - val_f1
            acc_gap = train_acc - val_acc

            f1_scores.append(val_f1)
            acc_scores.append(val_acc)
            f1_gaps.append(f1_gap)

            # Log with overfitting detection
            logger.info(f"  Fold {fold + 1} Results:")
            logger.info(f"    Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Gap: {f1_gap:.4f}")
            logger.info(
                f"    Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Gap: {acc_gap:.4f}"
            )
            logger.info(f"    Trees used: {fold_model.num_boosted_rounds()}")

            # CRITICAL: Flag overfitting
            if f1_gap > 0.15:  # More than 15% gap
                logger.error(f"    ⚠️ SEVERE OVERFITTING DETECTED! (F1 gap = {f1_gap:.1%})")
                logger.error(f"       → Reduce features or increase regularization")
            elif f1_gap > 0.10:
                logger.warning(f"    ⚠️ Moderate overfitting (F1 gap = {f1_gap:.1%})")
            else:
                logger.info(f"    ✓ Good generalization")

            if val_f1 > best_f1:
                best_f1 = val_f1
                # Save best booster for final model
                best_booster = fold_model


        logger.info(f"Random Forest CV F1-macro: {np.mean(rf_cv_f1):.4f} ± {np.std(rf_cv_f1):.4f}")
        logger.info(f"Random Forest CV Accuracy: {np.mean(rf_cv_acc):.4f} ± {np.std(rf_cv_acc):.4f}")

        # Compute average overfitting
        avg_f1_gap = np.mean(f1_gaps)
        logger.info(f"\nAverage train-val F1 gap across folds: {avg_f1_gap:.1%}")

        if avg_f1_gap > 0.12:
            logger.error("⚠️ MODEL IS OVERFITTING - Consider:")
            logger.error("  1. Reduce max_patterns (currently using too many)")
            logger.error("  2. Increase min_child_weight or reduce max_depth")
            logger.error("  3. Add more regularization (reg_alpha, reg_lambda)")

        # Final model: Train on full data with best params
        logger.info("\nTraining final model on full dataset...")
        dtrain_full = xgb.DMatrix(X_clean, label=y)
        final_booster = xgb.train(
            params=self.params,
            dtrain=dtrain_full,
            num_boost_round=500,
            verbose_eval=False,
        )

        # Convert best booster to XGBClassifier (for scikit-learn compatibility)
        final_model = xgb.XGBClassifier(**self.params)
        if best_booster:
            final_model._Booster = best_booster
        final_model.fit(X_clean, y)  # Fit to set sklearn attributes

        metrics = {
            "avg_f1_macro": np.mean(f1_scores),
            "std_f1_macro": np.std(f1_scores),
            "best_f1_macro": best_f1,
            "avg_accuracy": np.mean(acc_scores),
            "avg_f1_gap": avg_f1_gap,
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
        logger.info(f"   Avg Overfit Gap:   {avg_f1_gap:.1%}")
        logger.info(f"   Features used:     {X_clean.shape[1]} (numeric + contrast patterns)")
        logger.info(f"   Final trees:       {metrics['best_n_trees']}")
        logger.info("=" * 70)

        # Feature importance analysis
        feature_importance = pd.DataFrame(
            {"feature": X_clean.columns, "importance": final_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info("\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Save to file
        import os

        os.makedirs("output/latest_run", exist_ok=True)
        feature_importance.to_csv("output/latest_run/feature_importance.csv", index=False)

        return final_model, metrics
