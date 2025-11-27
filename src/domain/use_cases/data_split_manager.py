"""
DataSplitManager — Handles train/test/validation splitting with proper stratification.

Implements:
- Train/Test split (80/20 or configurable ratio)
- Stratified K-Fold cross-validation on training set
- Temporal split support for time-series data
- Proper index tracking for reproducibility
"""

import logging
from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split
from collections import Counter

logger = logging.getLogger(__name__)


class DataSplitManager:
    """Manages data splitting for training, validation, and testing."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        n_cv_splits: int = 5,
        temporal_split: bool = False,
    ):
        """
        Initialize DataSplitManager.

        Args:
            test_size: Proportion of data to hold out as test set (default: 0.2 = 80/20 split)
            random_state: Random seed for reproducibility
            n_cv_splits: Number of cross-validation folds on training set
            temporal_split: If True, use TimeSeriesSplit instead of StratifiedKFold
        """
        self.test_size = test_size
        self.random_state = random_state
        self.n_cv_splits = n_cv_splits
        self.temporal_split = temporal_split

        self.train_idx: Optional[np.ndarray] = None
        self.test_idx: Optional[np.ndarray] = None
        self.cv_splits: List[Tuple[np.ndarray, np.ndarray]] = []

    def split_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        class_labels: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets with stratification.

        Args:
            X: Feature matrix
            y: Target labels
            class_labels: Optional class label names for logging

        Returns:
            X_train, X_test, y_train, y_test
        """
        if len(X) != len(y):
            raise ValueError(
                f"Feature matrix and target misaligned! "
                f"X has {len(X)} rows but y has {len(y)} labels."
            )

        logger.info(f"Splitting {len(X)} samples into train ({1-self.test_size:.0%}) "
                   f"and test ({self.test_size:.0%}) sets")

        # Stratified split to maintain class distribution
        self.train_idx, self.test_idx = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        X_train = X.iloc[self.train_idx].reset_index(drop=True)
        X_test = X.iloc[self.test_idx].reset_index(drop=True)
        y_train = y[self.train_idx]
        y_test = y[self.test_idx]

        # Log split statistics
        self._log_split_stats(y_train, y_test, class_labels)

        return X_train, X_test, y_train, y_test

    def create_cv_splits(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        class_labels: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits on training data.

        Args:
            X_train: Training feature matrix
            y_train: Training target labels
            class_labels: Optional class label names for logging

        Returns:
            List of (train_idx, val_idx) tuples for each CV fold
        """
        logger.info(f"Creating {self.n_cv_splits}-fold cross-validation splits on training set")

        # Verify minimum class size for safe CV
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values())
        safe_n_splits = min(self.n_cv_splits, min_class_count // 2)

        if safe_n_splits < self.n_cv_splits:
            logger.warning(
                f"Reducing CV splits from {self.n_cv_splits} to {safe_n_splits} "
                f"due to small class sizes (min={min_class_count})"
            )
            safe_n_splits_local = safe_n_splits
        else:
            safe_n_splits_local = self.n_cv_splits

        if safe_n_splits_local < 3:
            raise ValueError(
                f"Insufficient data for reliable CV! Only {min_class_count} samples "
                f"in smallest class. Cannot create {self.n_cv_splits} CV folds."
            )

        if self.temporal_split:
            splitter = TimeSeriesSplit(n_splits=safe_n_splits_local)
            self.cv_splits = list(splitter.split(X_train))
            logger.info(f"Using TimeSeriesSplit for temporal data")
        else:
            splitter = StratifiedKFold(
                n_splits=safe_n_splits_local,
                shuffle=True,
                random_state=self.random_state,
            )
            self.cv_splits = list(splitter.split(X_train, y_train))
            logger.info(f"Using StratifiedKFold for balanced splits")

        self._log_cv_statistics(X_train, y_train, class_labels)

        return self.cv_splits

    def get_cv_fold(self, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Get data for a specific CV fold.

        Args:
            fold_idx: Index of the CV fold (0-indexed)

        Returns:
            X_train_fold, X_val_fold, y_train_fold, y_val_fold
        """
        if fold_idx >= len(self.cv_splits):
            raise ValueError(f"CV fold {fold_idx} does not exist (only {len(self.cv_splits)} folds)")

        train_fold_idx, val_fold_idx = self.cv_splits[fold_idx]

        # Note: These indices refer to the training set, so we need to map them appropriately
        # if X_train/y_train are passed separately
        return train_fold_idx, val_fold_idx

    def _log_split_stats(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        class_labels: Optional[np.ndarray] = None,
    ):
        """Log statistics about the train/test split."""
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)

        logger.info(f"✓ Train set: {len(y_train)} samples")
        for class_idx, count in sorted(train_counts.items()):
            label = class_labels[class_idx] if class_labels is not None else f"Class {class_idx}"
            logger.info(f"    {label}: {count} ({count/len(y_train)*100:.1f}%)")

        logger.info(f"✓ Test set: {len(y_test)} samples")
        for class_idx, count in sorted(test_counts.items()):
            label = class_labels[class_idx] if class_labels is not None else f"Class {class_idx}"
            logger.info(f"    {label}: {count} ({count/len(y_test)*100:.1f}%)")

    def _log_cv_statistics(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        class_labels: Optional[np.ndarray] = None,
    ):
        """Log statistics about CV splits."""
        logger.info(f"✓ CV Fold distribution:")
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splits):
            y_train_fold = y_train[train_idx]
            y_val_fold = y_train[val_idx]

            train_dist = Counter(y_train_fold)
            val_dist = Counter(y_val_fold)

            logger.info(f"  Fold {fold_idx + 1}:")
            logger.info(f"    Train: {len(train_idx)} | Val: {len(val_idx)}")
            logger.info(f"    Train dist: {dict(train_dist)}")
            logger.info(f"    Val dist:   {dict(val_dist)}")
