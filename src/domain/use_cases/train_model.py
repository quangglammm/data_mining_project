"""
TrainModelUseCase — Optimized training pipeline with train/test split, CV, testing, and checkpointing.

Features:
- Proper train/test split (80/20 by default)
- Cross-validation training on training set
- Comprehensive testing on held-out test set
- Checkpoint saving for all models
- Per-model performance tracking
"""

import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

# Model imports
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from lightgbm import LGBMClassifier

# Metrics imports
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# New imports for modular approach
from src.domain.use_cases.optimize_hyperparameters import tune_hyperparameters
from src.domain.use_cases.data_split_manager import DataSplitManager
from src.domain.use_cases.model_checkpoint_manager import ModelCheckpointManager
from src.domain.use_cases.model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)



class TrainModelUseCase:
    """
    Optimized model training with train/test split, cross-validation, testing, and checkpointing.
    
    Workflow:
    1. Split data into train (80%) and test (20%) sets
    2. Train multiple models on training set with cross-validation
    3. Evaluate each model on held-out test set
    4. Save model checkpoints with metrics and metadata
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        test_size: float = 0.2,
        checkpoint_dir: str = "models/checkpoints",
        test_results_dir: str = "output/latest_run/test_results",
    ):
        """
        Initialize TrainModelUseCase.

        Args:
            n_splits: Number of CV folds for training set
            random_state: Random seed for reproducibility
            test_size: Proportion of data to hold out as test set (default: 0.2)
            checkpoint_dir: Directory to save model checkpoints
            test_results_dir: Directory to save test results
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.test_size = test_size

        # Initialize managers
        self.data_splitter = DataSplitManager(
            test_size=test_size,
            random_state=random_state,
            n_cv_splits=n_splits,
            temporal_split=False,
        )
        self.checkpoint_manager = ModelCheckpointManager(base_checkpoint_dir=checkpoint_dir)
        self.evaluator = ModelEvaluator(output_dir=test_results_dir)

        # XGBoost parameters
        self.xgb_params = {
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute the complete training pipeline with train/test split, CV, testing, and checkpointing.

        Args:
            X: Feature matrix
            y: Target labels (class indices)
            class_labels: Array of class label names
            tune_hyperparams: Whether to perform hyperparameter tuning

        Returns:
            Tuple of (final_models_dict, all_metrics_dict)
        """
        logger.info("=" * 70)
        logger.info("STARTING OPTIMIZED TRAINING PIPELINE")
        logger.info("=" * 70)

        # ===== STEP 1: DATA VALIDATION AND PREPROCESSING =====
        logger.info("\n[STEP 1] Data Validation and Preprocessing")
        logger.info("-" * 70)

        if len(X) != len(y):
            raise ValueError(
                f"Feature matrix and target misaligned! "
                f"X has {len(X)} rows but y has {len(y)} labels."
            )

        logger.info(f"✓ Verified: X and y both have {len(X)} samples")

        # Clean data
        X = self._clean_data(X)

        # Check class distribution
        class_counts = Counter(y)
        logger.info(f"✓ Class distribution: {dict(class_counts)}")

        # ===== STEP 2: TRAIN/TEST SPLIT =====
        logger.info("\n[STEP 2] Train/Test Split")
        logger.info("-" * 70)

        X_train, X_test, y_train, y_test = self.data_splitter.split_data(
            X, y, class_labels
        )

        logger.info(f"✓ Train set: {len(X_train)} samples")
        logger.info(f"✓ Test set:  {len(X_test)} samples")

        # ===== STEP 3: CROSS-VALIDATION SETUP =====
        logger.info("\n[STEP 3] Cross-Validation Setup")
        logger.info("-" * 70)

        cv_splits = self.data_splitter.create_cv_splits(X_train, y_train, class_labels)
        logger.info(f"✓ Created {len(cv_splits)} CV folds for training set")

        # ===== STEP 4: TRAIN BASELINE MODELS =====
        logger.info("\n[STEP 4] Training Baseline Models with CV")
        logger.info("-" * 70)

        baseline_models = self._train_baseline_models(
            X_train, y_train, cv_splits, class_labels
        )

        # ===== STEP 5: TRAIN XGBOOST WITH CV =====
        logger.info("\n[STEP 5] Training XGBoost with CV and Native Early Stopping")
        logger.info("-" * 70)

        if tune_hyperparams:
            logger.info("Running hyperparameter tuning...")
            best_params = tune_hyperparameters(X_train, y_train, n_splits=3, n_iter=20)
            self.xgb_params.update(best_params)
            logger.info(f"✓ Using tuned parameters")

        xgb_model, xgb_cv_results = self._train_xgboost_with_cv(
            X_train, y_train, cv_splits, class_labels
        )

        # ===== STEP 6: TRAIN FINAL MODELS ON FULL TRAINING SET =====
        logger.info("\n[STEP 6] Training Final Models on Full Training Set")
        logger.info("-" * 70)

        final_models = self._train_final_models(
            X_train, y_train, X_test, y_test, xgb_model, baseline_models
        )

        # ===== STEP 7: TEST ALL MODELS ON HELD-OUT TEST SET =====
        logger.info("\n[STEP 7] Testing All Models on Held-Out Test Set")
        logger.info("-" * 70)

        test_results = self._test_all_models(
            final_models, X_test, y_test, class_labels
        )

        # ===== STEP 8: SAVE CHECKPOINTS =====
        logger.info("\n[STEP 8] Saving Model Checkpoints")
        logger.info("-" * 70)

        checkpoint_info = self._save_all_checkpoints(
            final_models,
            test_results,
            X_train.shape,
            X_test.shape,
            class_labels,
        )

        # ===== STEP 9: GENERATE REPORTS =====
        logger.info("\n[STEP 9] Generating Reports")
        logger.info("-" * 70)

        self.evaluator.save_test_results()
        report = self.evaluator.generate_test_report(
            output_file="output/latest_run/test_results/test_report.txt"
        )

        # ===== FINAL SUMMARY =====
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 70)

        # Prepare return data
        all_metrics = {
            "cv_results": xgb_cv_results,
            "test_results": test_results,
            "checkpoint_info": checkpoint_info,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "class_labels": class_labels.tolist() if isinstance(class_labels, np.ndarray) else class_labels,
        }

        logger.info(f"\n📊 Summary:")
        logger.info(f"   Training samples:   {len(X_train)}")
        logger.info(f"   Test samples:       {len(X_test)}")
        logger.info(f"   Features:           {X_train.shape[1]}")
        logger.info(f"   Models trained:     {len(final_models)}")
        logger.info(f"   Checkpoints saved:  {len(checkpoint_info)}")
        logger.info(f"   Results saved to:   output/latest_run/test_results/")
        logger.info("=" * 70)

        # Return best model as first element for backward compatibility
        best_model = final_models.get("xgboost", final_models.get(list(final_models.keys())[0]))
        return (best_model, all_metrics)

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling NaN and infinite values."""
        # Check for NaN values
        if X.isnull().any().any():
            n_nan = X.isnull().sum().sum()
            logger.warning(f"Found {n_nan} NaN values in features - filling with 0")
            X = X.fillna(0)

        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number]).values).any():
            logger.warning("Found infinite values in features - clipping")
            X = X.replace([np.inf, -np.inf], 0)

        # Drop year column if present
        if "year" in X.columns:
            X = X.drop(columns=["year"])
            logger.info("✓ Dropped 'year' column")

        return X

    def _train_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_splits: list,
        class_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Train baseline models with cross-validation."""
        baseline_results = {}

        # Dummy Classifier (random baseline)
        logger.info("\nTraining Dummy Classifier...")
        dummy_f1_scores = []
        dummy_acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            dummy = DummyClassifier(strategy='stratified', random_state=self.random_state)
            dummy.fit(X_train_fold, y_train_fold)

            y_pred = dummy.predict(X_val_fold)
            dummy_f1_scores.append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
            dummy_acc_scores.append(accuracy_score(y_val_fold, y_pred))

        baseline_results["dummy"] = {
            "f1_mean": np.mean(dummy_f1_scores),
            "f1_std": np.std(dummy_f1_scores),
            "accuracy_mean": np.mean(dummy_acc_scores),
            "accuracy_std": np.std(dummy_acc_scores),
        }
        logger.info(f"✓ Dummy F1-macro CV: {baseline_results['dummy']['f1_mean']:.4f} ± {baseline_results['dummy']['f1_std']:.4f}")

        # Random Forest
        logger.info("\nTraining Random Forest...")
        rf_f1_scores = []
        rf_acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=self.random_state)
            rf.fit(X_train_fold, y_train_fold)

            y_pred = rf.predict(X_val_fold)
            rf_f1_scores.append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
            rf_acc_scores.append(accuracy_score(y_val_fold, y_pred))

        baseline_results["random_forest"] = {
            "f1_mean": np.mean(rf_f1_scores),
            "f1_std": np.std(rf_f1_scores),
            "accuracy_mean": np.mean(rf_acc_scores),
            "accuracy_std": np.std(rf_acc_scores),
        }
        logger.info(f"✓ Random Forest F1-macro CV: {baseline_results['random_forest']['f1_mean']:.4f} ± {baseline_results['random_forest']['f1_std']:.4f}")

        # Logistic Regression
        logger.info("\nTraining Logistic Regression...")
        lr_f1_scores = []
        lr_acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            lr_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(solver='lbfgs', max_iter=500, C=1.0, random_state=self.random_state))
            ])
            lr_pipeline.fit(X_train_fold, y_train_fold)

            y_pred = lr_pipeline.predict(X_val_fold)
            lr_f1_scores.append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
            lr_acc_scores.append(accuracy_score(y_val_fold, y_pred))

        baseline_results["logistic_regression"] = {
            "f1_mean": np.mean(lr_f1_scores),
            "f1_std": np.std(lr_f1_scores),
            "accuracy_mean": np.mean(lr_acc_scores),
            "accuracy_std": np.std(lr_acc_scores),
        }
        logger.info(f"✓ Logistic Regression F1-macro CV: {baseline_results['logistic_regression']['f1_mean']:.4f} ± {baseline_results['logistic_regression']['f1_std']:.4f}")

        # KNN
        logger.info("\nTraining KNN...")
        knn_f1_scores = []
        knn_acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            knn_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
            ])
            knn_pipeline.fit(X_train_fold, y_train_fold)

            y_pred = knn_pipeline.predict(X_val_fold)
            knn_f1_scores.append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
            knn_acc_scores.append(accuracy_score(y_val_fold, y_pred))

        baseline_results["knn"] = {
            "f1_mean": np.mean(knn_f1_scores),
            "f1_std": np.std(knn_f1_scores),
            "accuracy_mean": np.mean(knn_acc_scores),
            "accuracy_std": np.std(knn_acc_scores),
        }
        logger.info(f"✓ KNN F1-macro CV: {baseline_results['knn']['f1_mean']:.4f} ± {baseline_results['knn']['f1_std']:.4f}")

        # SVM
        logger.info("\nTraining SVM...")
        svm_f1_scores = []
        svm_acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            svm_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=self.random_state))
            ])
            svm_pipeline.fit(X_train_fold, y_train_fold)

            y_pred = svm_pipeline.predict(X_val_fold)
            svm_f1_scores.append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
            svm_acc_scores.append(accuracy_score(y_val_fold, y_pred))

        baseline_results["svm"] = {
            "f1_mean": np.mean(svm_f1_scores),
            "f1_std": np.std(svm_f1_scores),
            "accuracy_mean": np.mean(svm_acc_scores),
            "accuracy_std": np.std(svm_acc_scores),
        }
        logger.info(f"✓ SVM F1-macro CV: {baseline_results['svm']['f1_mean']:.4f} ± {baseline_results['svm']['f1_std']:.4f}")

        # LightGBM
        logger.info("\nTraining LightGBM...")
        lgb_f1_scores = []
        lgb_acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            lgb = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=200, random_state=self.random_state, verbose=-1)
            lgb.fit(X_train_fold, y_train_fold)

            y_pred = lgb.predict(X_val_fold)
            lgb_f1_scores.append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
            lgb_acc_scores.append(accuracy_score(y_val_fold, y_pred))

        baseline_results["lightgbm"] = {
            "f1_mean": np.mean(lgb_f1_scores),
            "f1_std": np.std(lgb_f1_scores),
            "accuracy_mean": np.mean(lgb_acc_scores),
            "accuracy_std": np.std(lgb_acc_scores),
        }
        logger.info(f"✓ LightGBM F1-macro CV: {baseline_results['lightgbm']['f1_mean']:.4f} ± {baseline_results['lightgbm']['f1_std']:.4f}")

        # MLP (Neural Network)
        logger.info("\nTraining MLP...")
        mlp_f1_scores = []
        mlp_acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            mlp_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=0.01, random_state=self.random_state))
            ])
            mlp_pipeline.fit(X_train_fold, y_train_fold)

            y_pred = mlp_pipeline.predict(X_val_fold)
            mlp_f1_scores.append(f1_score(y_val_fold, y_pred, average="macro", zero_division=0))
            mlp_acc_scores.append(accuracy_score(y_val_fold, y_pred))

        baseline_results["mlp"] = {
            "f1_mean": np.mean(mlp_f1_scores),
            "f1_std": np.std(mlp_f1_scores),
            "accuracy_mean": np.mean(mlp_acc_scores),
            "accuracy_std": np.std(mlp_acc_scores),
        }
        logger.info(f"✓ MLP F1-macro CV: {baseline_results['mlp']['f1_mean']:.4f} ± {baseline_results['mlp']['f1_std']:.4f}")

        logger.info(f"\n✅ Baseline models trained: {list(baseline_results.keys())}")

        return baseline_results

    def _train_xgboost_with_cv(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cv_splits: list,
        class_labels: np.ndarray,
    ) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
        """Train XGBoost with cross-validation and native early stopping."""
        logger.info("Training XGBoost with cross-validation...")

        xgb_f1_scores = []
        xgb_acc_scores = []
        xgb_gaps = []
        best_f1 = 0.0
        best_booster = None

        cv_results = {
            "fold_f1_scores": [],
            "fold_accuracy_scores": [],
            "fold_train_f1_scores": [],
            "fold_val_f1_scores": [],
            "fold_overfitting_gaps": [],
        }

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info(f"\n  Fold {fold + 1}/{len(cv_splits)}")

            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

            # Train with native early stopping
            booster = xgb.train(
                params=self.xgb_params,
                dtrain=dtrain,
                num_boost_round=500,
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )

            # Predict
            y_train_pred = booster.predict(dtrain).argmax(axis=1)
            y_val_pred = booster.predict(dval).argmax(axis=1)

            # Calculate metrics
            train_f1 = f1_score(y_train_fold, y_train_pred, average="macro", zero_division=0)
            val_f1 = f1_score(y_val_fold, y_val_pred, average="macro", zero_division=0)
            val_acc = accuracy_score(y_val_fold, y_val_pred)
            f1_gap = train_f1 - val_f1

            xgb_f1_scores.append(val_f1)
            xgb_acc_scores.append(val_acc)
            xgb_gaps.append(f1_gap)

            cv_results["fold_f1_scores"].append(val_f1)
            cv_results["fold_accuracy_scores"].append(val_acc)
            cv_results["fold_train_f1_scores"].append(train_f1)
            cv_results["fold_val_f1_scores"].append(val_f1)
            cv_results["fold_overfitting_gaps"].append(f1_gap)

            logger.info(f"    Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Gap: {f1_gap:.4f}")
            logger.info(f"    Trees: {booster.num_boosted_rounds()}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_booster = booster

        # Calculate aggregate CV metrics
        avg_f1 = np.mean(xgb_f1_scores)
        std_f1 = np.std(xgb_f1_scores)
        avg_acc = np.mean(xgb_acc_scores)
        avg_gap = np.mean(xgb_gaps)

        cv_results.update({
            "avg_f1_macro": avg_f1,
            "std_f1_macro": std_f1,
            "best_f1_macro": best_f1,
            "avg_accuracy": avg_acc,
            "avg_overfitting_gap": avg_gap,
        })

        logger.info(f"\n  ✓ XGBoost CV Results:")
        logger.info(f"    Avg F1-macro: {avg_f1:.4f} ± {std_f1:.4f}")
        logger.info(f"    Avg Accuracy: {avg_acc:.4f}")
        logger.info(f"    Avg Overfit Gap: {avg_gap:.1%}")

        # Train final XGBoost model on full training set
        logger.info("\n  Training final XGBoost on full training set...")
        dtrain_full = xgb.DMatrix(X_train, label=y_train)
        final_booster = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain_full,
            num_boost_round=500,
            verbose_eval=False,
        )

        final_model = xgb.XGBClassifier(**self.xgb_params)
        final_model.fit(X_train, y_train)

        return final_model, cv_results

    def _train_final_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        xgb_model: xgb.XGBClassifier,
        baseline_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train final models on full training set."""
        logger.info("Training final models on full training set...")

        final_models = {}

        # XGBoost (already trained above)
        final_models["xgboost"] = xgb_model

        # Random Forest
        logger.info("  Training Random Forest...")
        rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=self.random_state)
        rf.fit(X_train, y_train)
        final_models["random_forest"] = rf

        # Logistic Regression
        logger.info("  Training Logistic Regression...")
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(solver='lbfgs', max_iter=500, C=1.0, random_state=self.random_state))
        ])
        lr_pipeline.fit(X_train, y_train)
        final_models["logistic_regression"] = lr_pipeline

        # KNN
        logger.info("  Training KNN...")
        knn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
        ])
        knn_pipeline.fit(X_train, y_train)
        final_models["knn"] = knn_pipeline

        # SVM
        logger.info("  Training SVM...")
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=self.random_state))
        ])
        svm_pipeline.fit(X_train, y_train)
        final_models["svm"] = svm_pipeline

        # LightGBM
        logger.info("  Training LightGBM...")
        lgb = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=200, random_state=self.random_state, verbose=-1)
        lgb.fit(X_train, y_train)
        final_models["lightgbm"] = lgb

        # MLP
        logger.info("  Training MLP...")
        mlp_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=0.01, random_state=self.random_state))
        ])
        mlp_pipeline.fit(X_train, y_train)
        final_models["mlp"] = mlp_pipeline

        logger.info(f"\n✅ Final models trained: {list(final_models.keys())}")

        return final_models

    def _test_all_models(
        self,
        final_models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        class_labels: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Test all models on held-out test set."""
        logger.info("\n" + "=" * 70)
        logger.info("COMPREHENSIVE TESTING ON HELD-OUT TEST SET")
        logger.info("=" * 70)

        test_results = {}

        for model_name, model in final_models.items():
            metrics = self.evaluator.evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name,
                class_labels=class_labels,
            )
            test_results[model_name] = metrics

        return test_results

    def _save_all_checkpoints(
        self,
        final_models: Dict[str, Any],
        test_results: Dict[str, Dict[str, Any]],
        train_shape: Tuple[int, int],
        test_shape: Tuple[int, int],
        class_labels: np.ndarray,
    ) -> Dict[str, Path]:
        """Save checkpoints for all models."""
        checkpoint_info = {}

        for model_name, model in final_models.items():
            metrics = test_results.get(model_name, {})

            train_stats = {
                "n_samples": train_shape[0],
                "n_features": train_shape[1],
                "test_samples": test_shape[0],
            }

            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                X_cols = list(range(train_shape[1]))
                feature_importance = pd.DataFrame({
                    "feature": [f"feature_{i}" for i in X_cols],
                    "importance": model.feature_importances_
                }).sort_values("importance", ascending=False)
            elif hasattr(model, 'named_steps') and 'lr' in model.named_steps:
                # Handle pipeline models
                try:
                    X_cols = list(range(train_shape[1]))
                    feature_importance = pd.DataFrame({
                        "feature": [f"feature_{i}" for i in X_cols],
                        "importance": np.abs(model.named_steps['lr'].coef_[0])
                    }).sort_values("importance", ascending=False)
                except:
                    pass

            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_model_checkpoint(
                model=model,
                model_name=model_name,
                metrics={
                    "accuracy": metrics.get("accuracy", 0),
                    "f1_macro": metrics.get("f1_macro", 0),
                    "f1_weighted": metrics.get("f1_weighted", 0),
                    "precision_macro": metrics.get("precision_macro", 0),
                    "recall_macro": metrics.get("recall_macro", 0),
                },
                cv_results=None,
                feature_importance=feature_importance,
                hyperparams=self.xgb_params if model_name == "xgboost" else None,
                train_stats=train_stats,
            )

            checkpoint_info[model_name] = checkpoint_path

        logger.info(f"\n✅ All checkpoints saved to: {self.checkpoint_manager.base_dir}")

        return {k: str(v) for k, v in checkpoint_info.items()}

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
                ('lr', LogisticRegression(solver='lbfgs', max_iter=500, C=1.0, random_state=42))
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
                ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=0.01, random_state=42))
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