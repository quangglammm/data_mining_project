"""Use case for building feature matrix."""

import logging
from typing import Tuple, Set, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class BuildFeatureMatrixUseCase:
    """Use case to build feature matrix from aggregated and pattern data."""

    def execute(
        self,
        df_agg: pd.DataFrame,
        df_sequences: pd.DataFrame,
        all_patterns: Set[Tuple[str, ...]],
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray]:
        """
        Execute feature matrix construction.

        Args:
            df_agg: DataFrame with aggregated numeric features
            df_sequences: DataFrame with event sequences
            all_patterns: Set of all unique patterns

        Returns:
            Tuple of (X, y, feature_names, class_labels)
        """
        logger.info(f"Building feature matrix with {len(all_patterns)} patterns")

        # Build pattern features
        sequences_as_sets = df_sequences["event_sequence"].apply(set).tolist()
        pattern_features = []

        for pattern in all_patterns:
            pattern_set = set(pattern)
            col_name = f"pat_{'__'.join(pattern)}"
            feature_col = [
                1 if pattern_set.issubset(seq_set) else 0
                for seq_set in sequences_as_sets
            ]
            pattern_features.append(pd.Series(feature_col, name=col_name))

        df_patterns = pd.concat(pattern_features, axis=1)

        # Build numeric features
        df_numeric = df_agg.drop(columns=["id_vụ", "year", "yield_class"])
        df_numeric = df_numeric.fillna(0)

        # Align indices
        df_numeric.index = df_sequences.index
        df_patterns.index = df_sequences.index

        # Combine features
        X = pd.concat([df_numeric, df_patterns], axis=1)

        # Get target
        df_target = df_sequences[["id_vụ", "year", "yield_class"]]

        # Sort by year
        df_final = pd.concat([X, df_target], axis=1)
        df_final = df_final.sort_values(by="year")
        df_final = df_final.reset_index(drop=True)

        # Separate X and y
        y_labels = df_final["yield_class"]
        X_sorted = df_final.drop(columns=["id_vụ", "yield_class"])

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)

        logger.info(f"Feature matrix shape: {X_sorted.shape}")
        return X_sorted, y_encoded, X_sorted.columns.tolist(), le.classes_

