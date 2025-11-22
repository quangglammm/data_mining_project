# src/domain/use_cases/mine_contrast_patterns.py
# FINAL BATTLE-TESTED VERSION — SEQUENTIAL + FAST + BEAUTIFUL OUTPUT

import os
import pandas as pd
from typing import Set, Tuple, List
import logging

logger = logging.getLogger(__name__)


def is_subsequence(pattern: Tuple[str, ...], sequence: List[str]) -> bool:
    """
    Fast subsequence check: does pattern appear in order (gaps allowed)?
    Example: ("A", "B") in ["X", "A", "Y", "B", "Z"] → True
    """
    if not pattern:
        return True
    pat_iter = iter(pattern)
    target = next(pat_iter)
    for event in sequence:
        if event == target:
            try:
                target = next(pat_iter)
            except StopIteration:
                return True
    return False


class MineContrastPatternsUseCase:
    def __init__(
        self,
        growth_high: float = 2.0,
        growth_low: float = 2.2,
        min_support_target: float = 0.05,  # 5% in target class
        min_global_support: float = 0.01,  # 1% overall
    ):
        self.growth_high = growth_high
        self.growth_low = growth_low
        self.min_support_target = min_support_target
        self.min_global_support = min_global_support

    def execute(
        self,
        df_sequences: pd.DataFrame,
        frequent_patterns: Set[Tuple[str, ...]],
        output_dir: str,
    ) -> pd.DataFrame:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Starting contrast pattern mining (High-yield & Low-yield risk patterns)")

        # Masks and counts
        high_mask = df_sequences["yield_class"] == "High"
        low_mask = df_sequences["yield_class"] == "Low"
        n_high = high_mask.sum()
        n_low = low_mask.sum()
        n_total = len(df_sequences)

        if n_high == 0 or n_low == 0:
            logger.warning("No High or Low seasons found!")
            return pd.DataFrame()

        # Extract raw lists (NO set conversion!)
        high_seqs = [seq for seq in df_sequences.loc[high_mask, "event_sequence"].tolist() if seq]
        low_seqs = [seq for seq in df_sequences.loc[low_mask, "event_sequence"].tolist() if seq]

        results = []
        epsilon = 1e-8

        logger.info(
            f"Scanning {len(frequent_patterns)} candidate patterns with sequential matching..."
        )

        for pat in frequent_patterns:
            if len(pat) < 2:  # skip single events
                continue

            # Sequential count (this is the magic!)
            in_high = sum(is_subsequence(pat, seq) for seq in high_seqs)
            in_low = sum(is_subsequence(pat, seq) for seq in low_seqs)

            supp_high = in_high / n_high
            supp_low = in_low / n_low
            supp_global = (in_high + in_low) / n_total

            if supp_global < self.min_global_support:
                continue

            pattern_str = " → ".join(pat)

            # High-yield marker
            if supp_high >= self.min_support_target:
                growth = supp_high / (supp_low + epsilon)
                if growth >= self.growth_high:
                    results.append(
                        {
                            "events": pat,
                            "pattern_str": pattern_str,
                            "type": "High-yield marker",
                            "support_high_%": round(supp_high * 100, 2),
                            "support_low_%": round(supp_low * 100, 2),
                            "growth": round(growth, 2),
                        }
                    )

            # Low-yield risk
            if supp_low >= self.min_support_target:
                growth = supp_low / (supp_high + epsilon)
                if growth >= self.growth_low:
                    results.append(
                        {
                            "events": pat,
                            "pattern_str": pattern_str,
                            "type": "Low-yield risk",
                            "support_low_%": round(supp_low * 100, 2),
                            "support_high_%": round(supp_high * 100, 2),
                            "growth": round(growth, 2),
                        }
                    )

        # Final DataFrame + sorting
        contrast_df = pd.DataFrame(results)
        if not contrast_df.empty:
            contrast_df = contrast_df.sort_values(
                by=["type", "growth"], ascending=[True, False]
            ).reset_index(drop=True)

        self._save_contrast_report(contrast_df, output_dir)

        high_count = (
            len(contrast_df[contrast_df["type"] == "High-yield marker"])
            if not contrast_df.empty
            else 0
        )
        low_count = (
            len(contrast_df[contrast_df["type"] == "Low-yield risk"])
            if not contrast_df.empty
            else 0
        )

        logger.info(
            f"Contrast mining completed: {high_count} High-yield markers + {low_count} Low-yield risk patterns"
        )
        return contrast_df

    def _save_contrast_report(self, df: pd.DataFrame, output_dir: str) -> None:
        path = os.path.join(output_dir, "contrast_patterns.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("CONTRAST WEATHER PATTERNS FOR RICE YIELD (Mekong Delta)\n")
            f.write(f"Growth threshold: High ≥{self.growth_high}× | Low ≥{self.growth_low}×\n")
            f.write(f"Min support in target class: ≥{self.min_support_target:.1%}\n")
            f.write("=" * 95 + "\n\n")

            if df.empty:
                f.write("No contrast patterns found. Try lowering thresholds.\n")
                return

            for typ, title in [
                ("High-yield marker", "HIGH-YIELD MARKERS — Golden sequences for top yield"),
                ("Low-yield risk", "LOW-YIELD RISK PATTERNS — Avoid these!"),
            ]:
                subset = df[df["type"] == typ]
                if subset.empty:
                    continue
                f.write(f"{title}\n")
                f.write("-" * 80 + "\n")
                for _, row in subset.iterrows():
                    f.write(f"• {row['pattern_str']}\n")
                    if typ == "High-yield marker":
                        f.write(
                            f"   High: {row['support_high_%']}% | Low: {row['support_low_%']}% "
                            f"| Growth: {row['growth']}×\n"
                            f"   → Ideal weather progression\n\n"
                        )
                    else:
                        f.write(
                            f"   Low: {row['support_low_%']}% | High: {row['support_high_%']}% "
                            f"| Growth: {row['growth']}×\n"
                            f"   → High risk of disease / poor filling\n\n"
                        )
        logger.info(f"Contrast report saved → {path}")
