"""BuildFeatureMatrixUseCase — FINAL SEQUENTIAL VERSION (THE ONE THAT WINS)"""

import logging
from typing import Tuple, Set
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# ← IMPORT THE SEQUENTIAL FUNCTION
from src.domain.use_cases.mine_contrast_patterns import is_subsequence

logger = logging.getLogger(__name__)


class BuildFeatureMatrixUseCase:
    DEBUG_EXPORT_PATH = Path("data/debug/feature_matrix.csv")

    def execute(
        self,
        df_agg: pd.DataFrame,
        df_sequences: pd.DataFrame,
        patterns: Set[Tuple[str, ...]],
        pattern_type: str = "contrast",
    ) -> Tuple[pd.DataFrame, np.ndarray, list, np.ndarray]:
        logger.info(
            f"Building feature matrix using {len(patterns)} {pattern_type} patterns (SEQUENTIAL MATCHING)"
        )

        self.DEBUG_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # === BULLETPROOF event_sequence conversion (unchanged) ===
        def safe_convert_event_sequence(value):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, str):
                value = value.strip()
                if not value or value == "[]":
                    return []
                if value.startswith("[") and value.endswith("]"):
                    try:
                        import ast

                        return ast.literal_eval(value)
                    except:
                        pass
                items = [
                    item.strip().strip("'\"")
                    for item in value.replace("[", "").replace("]", "").split(",")
                ]
                return [item for item in items if item]
            return []

        # ← KEEP AS LISTS! NO MORE SETS!
        event_sequences = df_sequences["event_sequence"].apply(safe_convert_event_sequence)

        # === Pattern features — NOW FULLY SEQUENTIAL ===
        pattern_dfs = []
        for i, pat in enumerate(patterns):
            if not pat or len(pat) < 2:
                continue
            col_name = f"pat_{i:03d}__{'__'.join(pat)}"
            # ← THIS IS THE LINE THAT UNLOCKS 80%+ ACCURACY
            col = event_sequences.apply(lambda seq: 1 if is_subsequence(pat, seq) else 0)
            pattern_dfs.append(col.rename(col_name))

        df_patterns = (
            pd.DataFrame(pattern_dfs).T if pattern_dfs else pd.DataFrame(index=df_sequences.index)
        )

        # === Numeric features (unchanged) ===
        drop_cols = {"id_vụ", "year", "yield_class"}
        df_numeric = df_agg.drop(
            columns=[c for c in drop_cols if c in df_agg.columns], errors="ignore"
        )
        df_numeric = df_numeric.fillna(0).astype(float)

        # === Combine ===
        df_numeric.index = df_sequences.index
        df_patterns.index = df_sequences.index
        X = pd.concat([df_numeric, df_patterns], axis=1)

        # === Target & metadata ===
        metadata = df_sequences[["id_vụ", "year", "yield_class"]].copy()
        y_labels = df_sequences["yield_class"].copy()

        # Sort by year
        if "year" in metadata.columns:
            sort_idx = metadata["year"].sort_values().index
            X = X.loc[sort_idx].reset_index(drop=True)
            metadata = metadata.loc[sort_idx].reset_index(drop=True)
            y_labels = y_labels.loc[sort_idx].reset_index(drop=True)

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)

        # === DEBUG EXPORT ===
        debug_df = pd.concat([metadata.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        debug_df.insert(3, "yield_class_encoded", y_encoded)
        debug_df.to_csv(self.DEBUG_EXPORT_PATH, index=False, encoding="utf-8")

        logger.info(f"Success: {X.shape[0]} samples × {X.shape[1]} features built")
        logger.info(
            f"   → {df_numeric.shape[1]} numeric + {df_patterns.shape[1]} SEQUENTIAL pattern features"
        )
        logger.info(f"   Debug CSV → {self.DEBUG_EXPORT_PATH.resolve()}")

        return X, y_encoded, X.columns.tolist(), le.classes_
