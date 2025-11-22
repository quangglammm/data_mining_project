# src/domain/use_cases/mine_sequential_patterns.py
"""
Mine frequent sequential patterns from weather event sequences using PrefixSpan.
Responsible ONLY for discovering frequent patterns per yield class and saving results.
"""

import logging
from typing import Set, Tuple, List
import pandas as pd
from prefixspan import PrefixSpan
from pathlib import Path

logger = logging.getLogger(__name__)


class MineSequentialPatternsUseCase:
    """
    Extracts frequent sequential patterns separately for Low, Medium, and High-yield seasons.
    Outputs human-readable TXT reports and CSV files for analysis.
    """

    def __init__(
        self,
        min_support: float = 0.08,
        minlen: int = 2,
        maxlen: int = 5,
    ):
        """
        Args:
            min_support: Minimum support threshold (fraction of sequences in a class)
            minlen: Minimum pattern length
            maxlen: Maximum pattern length
        """
        self.min_support = min_support
        self.minlen = minlen
        self.maxlen = maxlen

    def execute(
        self,
        df_sequences: pd.DataFrame,
        output_dir: str = "output/latest_run/frequent",
    ) -> Set[Tuple[str, ...]]:
        """
        Mine frequent patterns per yield class and return the union of all patterns.

        Returns:
            Set of unique frequent patterns (as tuples) across all classes.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting frequent pattern mining (min_support={self.min_support:.1%})")

        all_patterns: Set[Tuple[str, ...]] = set()
        groups = df_sequences.groupby("yield_class")

        for yield_class, group in groups:
            sequences: List[List[str]] = group["event_sequence"].tolist()
            if not sequences:
                logger.warning(f"No sequences found for class '{yield_class}'")
                continue

            n_sequences = len(sequences)
            min_count = max(1, int(n_sequences * self.min_support))

            # Run PrefixSpan
            raw_patterns = PrefixSpan(sequences).frequent(min_count)

            # Filter by length
            filtered = [
                (tuple(pattern), freq)
                for freq, pattern in raw_patterns
                if self.minlen <= len(pattern) <= self.maxlen
            ]
            filtered_sorted = sorted(filtered, key=lambda x: (-x[1], len(x[0]), x[0]))

            # Save per-class results
            self._save_class_results(
                yield_class=yield_class,
                patterns=filtered_sorted,
                n_total=n_sequences,
                output_path=output_path,
            )

            # Update global set
            all_patterns.update(p[0] for p in filtered_sorted)

        # Save master list of all unique frequent patterns
        self._save_master_list(all_patterns, output_path)

        logger.info(f"Frequent mining completed: {len(all_patterns)} unique patterns discovered")
        return all_patterns

    def _save_class_results(
        self,
        yield_class: str,
        patterns: List[Tuple[Tuple[str, ...], int]],
        n_total: int,
        output_path: Path,
    ) -> None:
        """Save TXT (human-readable) and CSV results for one class."""
        df = pd.DataFrame(patterns, columns=["pattern", "frequency"])
        df["support_%"] = df["frequency"] / n_total * 100

        csv_file = output_path / f"frequent_patterns_{yield_class}.csv"
        txt_file = output_path / f"frequent_patterns_{yield_class}.txt"

        df.to_csv(csv_file, index=False, encoding="utf-8")

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(f"FREQUENT PATTERNS — {yield_class.upper()} YIELD SEASONS\n")
            f.write(f"Total seasons: {n_total} | Min support: {self.min_support:.1%}\n")
            f.write(f"Pattern length: {self.minlen}–{self.maxlen}\n")
            f.write("=" * 80 + "\n\n")
            for i, (pat, freq) in enumerate(patterns[:100], 1):
                support_pct = freq / n_total * 100
                f.write(f"{i:2}. {' → '.join(pat)}  (n={freq}, {support_pct:5.2f}%)\n")

        logger.info(f"Saved {len(patterns)} patterns for {yield_class} → {txt_file.name}")

    def _save_master_list(self, patterns: Set[Tuple[str, ...]], output_path: Path) -> None:
        """Save a clean list of all unique frequent patterns."""
        sorted_patterns = sorted(patterns, key=lambda x: (len(x), x))
        master_file = output_path / "all_frequent_patterns.txt"

        with open(master_file, "w", encoding="utf-8") as f:
            f.write(f"ALL UNIQUE FREQUENT PATTERNS (Total: {len(sorted_patterns)})\n")
            f.write(f"Min support: {self.min_support:.1%} | Length: {self.minlen}–{self.maxlen}\n")
            f.write("=" * 80 + "\n\n")
            for pat in sorted_patterns:
                f.write(" → ".join(pat) + "\n")
