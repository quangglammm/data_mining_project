"""Use case for mining sequential patterns."""

import logging
from typing import Set, Tuple, List
import pandas as pd
from prefixspan import PrefixSpan

logger = logging.getLogger(__name__)


class MineSequentialPatternsUseCase:
    """Use case to mine sequential patterns from event sequences."""

    def __init__(self, min_support: float = 0.1, minlen: int = 2, maxlen: int = 5):
        """
        Initialize use case.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
            minlen: Minimum pattern length
            maxlen: Maximum pattern length
        """
        self.min_support = min_support
        self.minlen = minlen
        self.maxlen = maxlen

    def execute(
        self, df_sequences: pd.DataFrame, yield_class_column: str = "yield_class"
    ) -> Set[Tuple[str, ...]]:
        """
        Execute pattern mining.

        Args:
            df_sequences: DataFrame with columns:
                - yield_class: yield class label
                - event_sequence: list of event strings
            yield_class_column: Name of yield class column

        Returns:
            Set of unique patterns (as tuples)
        """
        logger.info(
            f"Mining patterns: min_support={self.min_support}, "
            f"minlen={self.minlen}, maxlen={self.maxlen}"
        )

        # Split by yield class
        db_low = df_sequences[df_sequences[yield_class_column] == "Low"][
            "event_sequence"
        ].tolist()
        db_medium = df_sequences[df_sequences[yield_class_column] == "Medium"][
            "event_sequence"
        ].tolist()
        db_high = df_sequences[df_sequences[yield_class_column] == "High"][
            "event_sequence"
        ].tolist()

        logger.info(
            f"Sequences: Low={len(db_low)}, Medium={len(db_medium)}, High={len(db_high)}"
        )

        master_pattern_set = set()

        def find_and_filter(db: List[List[str]], support_percent: float, name: str):
            """Find patterns for a yield class."""
            if not db:
                logger.warning(f"No data for class '{name}', skipping")
                return []

            support_count = int(len(db) * support_percent)
            logger.info(
                f"Finding patterns for '{name}' "
                f"(min_support={support_percent * 100}%, count>={support_count})"
            )

            ps = PrefixSpan(db)
            patterns_gen = ps.frequent(support_count)

            filtered_patterns = [
                tuple(pat)
                for freq, pat in patterns_gen
                if self.minlen <= len(pat) <= self.maxlen
            ]

            top_5 = sorted(filtered_patterns, key=len, reverse=True)[:5]
            logger.info(f"Top 5 patterns for '{name}': {top_5}")

            return filtered_patterns

        # Mine patterns for each class
        patterns_low = find_and_filter(db_low, self.min_support, "Low")
        patterns_medium = find_and_filter(db_medium, self.min_support, "Medium")
        patterns_high = find_and_filter(db_high, self.min_support, "High")

        # Collect unique patterns
        master_pattern_set.update(patterns_low)
        master_pattern_set.update(patterns_medium)
        master_pattern_set.update(patterns_high)

        logger.info(f"Mined {len(master_pattern_set)} unique patterns")
        return master_pattern_set

