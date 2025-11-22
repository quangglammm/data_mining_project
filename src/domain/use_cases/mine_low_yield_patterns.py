# src/domain/use_cases/mine_low_yield_patterns.py
from typing import List, Tuple, Dict, Set, Any
import pandas as pd
import numpy as np
from collections import Counter
from prefixspan import PrefixSpan
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def is_subsequence(pattern: Tuple[str, ...], sequence: List[str]) -> bool:
    """Fast subsequence check (gaps allowed)."""
    if not pattern:
        return True
    it = iter(pattern)
    target = next(it)
    for event in sequence:
        if event == target:
            try:
                target = next(it)
            except StopIteration:
                return True
    return False


class MineLowYieldPatternsUseCase:
    def __init__(
        self,
        min_support_high: float = 0.05,
        min_support_low: float = 0.10,
        growth_threshold: float = 3.0,
        rare_max_support: float = 0.03,  # ≤3% trong Low
        min_drop_impact: float = 0.4,  # khi thấy → xác suất High giảm ≥40%
    ):
        self.min_support_high = min_support_high
        self.min_support_low = min_support_low
        self.growth_threshold = growth_threshold
        self.rare_max_support = rare_max_support
        self.min_drop_impact = min_drop_impact

    def execute(
        self,
        df_sequences: pd.DataFrame,
        high_golden_patterns: Set[Tuple[str, ...]],
        output_dir: str = "output/latest_run/destructive",
    ) -> Dict[str, Any]:
        """
        Trả về dict đơn giản + tự động lưu CSV + JSON
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        high_mask = df_sequences["yield_class"] == "High"
        low_mask = df_sequences["yield_class"] == "Low"
        high_seqs = df_sequences[high_mask]["event_sequence"].apply(self._to_list).tolist()
        low_seqs = df_sequences[low_mask]["event_sequence"].apply(self._to_list).tolist()
        n_high, n_low = len(high_seqs), len(low_seqs)

        # Containers cuối cùng (chỉ list, không object phức tạp)
        contrast_events = []  # ["stage_4_Đạo-ôn", "stage_5_Nắng-nóng"]
        destructive_patterns = []  # [("Mát-Khô", "Mưa-cực-lớn"), ...]
        breaker_events = []  # ["Nóng-Khô", "Mưa-lớn"]

        # =================================================================
        # 1. Contrast Events (sự kiện đơn lẻ phá hoại)
        # =================================================================
        event_counts_high = Counter(e for seq in high_seqs for e in seq)
        event_counts_low = Counter(e for seq in low_seqs for e in seq)

        contrast_report = []
        for event, cnt_low in event_counts_low.items():
            supp_high = event_counts_high[event] / n_high
            supp_low = cnt_low / n_low
            if supp_low >= self.min_support_low and supp_high <= self.min_support_high:
                growth = supp_low / (supp_high + 1e-8)
                if growth >= self.growth_threshold:
                    contrast_events.append(event)
                    contrast_report.append(
                        {
                            "event": event,
                            "support_low_%": round(supp_low * 100, 2),
                            "support_high_%": round(supp_high * 100, 2),
                            "growth": round(growth, 2),
                        }
                    )

        # =================================================================
        # 2. Rare but Catastrophic Patterns
        # =================================================================
        low_seqs_clean = [s for s in low_seqs if len(s) >= 2]
        rare_freq = int(self.rare_max_support * n_low)
        rare_results = PrefixSpan(low_seqs_clean).frequent(rare_freq)

        rare_report = []
        for freq, pat in rare_results:
            if len(pat) < 2:
                continue
            pat_tuple = tuple(pat)
            count_high = sum(1 for seq in high_seqs if is_subsequence(pat_tuple, seq))
            prob_high_if_seen = count_high / freq if freq > 0 else 0
            drop = 1 - prob_high_if_seen
            if drop >= self.min_drop_impact:
                destructive_patterns.append(pat_tuple)
                rare_report.append(
                    {
                        "pattern": " → ".join(pat),
                        "freq_in_low": freq,
                        "support_low_%": round(freq / n_low * 100, 2),
                        "prob_high_if_seen": round(prob_high_if_seen, 3),
                        "yield_drop": round(drop, 3),
                    }
                )

        # =================================================================
        # 3. Golden Sequence Breakers
        # =================================================================
        breaker_report = []
        for golden in high_golden_patterns:
            if len(golden) < 3:
                continue
            # Tìm các mùa Low có prefix của golden
            breakers = []
            for seq in low_seqs:
                breaker = self._find_first_breaker(seq, golden)
                if breaker:
                    breakers.append(breaker)

            if breakers:
                top_breaker, count = Counter(breakers).most_common(1)[0]
                breaker_events.append(top_breaker)
                breaker_report.append(
                    {
                        "golden_pattern": " → ".join(golden),
                        "broken_by": top_breaker,
                        "times_broken": count,
                        "percentage": round(count / len(breakers) * 100, 1),
                    }
                )

        # =================================================================
        # LƯU TẤT CẢ RA FILE (CSV + JSON + TXT đẹp)
        # =================================================================
        self._save_results(
            output_dir=output_dir,
            contrast_report=contrast_report,
            rare_report=rare_report,
            breaker_report=breaker_report,
        )

        # Trả về dict cực kỳ đơn giản để tích hợp
        return {
            "contrast_events": contrast_events,  # list[str]
            "destructive_patterns": destructive_patterns,  # list[tuple[str]]
            "breaker_events": breaker_events,  # list[str]
            "all_destructive_patterns": destructive_patterns
            + [(e,) for e in contrast_events + breaker_events],
            "reports": {  # để hiển thị
                "contrast_events": contrast_report,
                "rare_destructive": rare_report,
                "golden_breakers": breaker_report,
            },
        }

    # =================================================================
    # Helper functions
    # =================================================================
    def _to_list(self, x):
        if isinstance(x, (list, tuple)):
            return list(x)
        if pd.isna(x):
            return []
        try:
            import ast

            return ast.literal_eval(x) if isinstance(x, str) else []
        except:
            return []

    def _find_first_breaker(self, sequence: List[str], golden: Tuple[str, ...]) -> str | None:
        """Tìm sự kiện đầu tiên phá vỡ chuỗi golden."""
        pos = 0
        for event in golden:
            try:
                idx = sequence.index(event, pos)
                pos = idx + 1
            except ValueError:
                return None
        return sequence[pos] if pos < len(sequence) else None

    def _save_results(self, output_dir: str, contrast_report, rare_report, breaker_report):
        # 1. Beautiful TXT report
        txt_path = Path(output_dir) / "LOW_YIELD_DESTRUCTIVE_MECHANISMS.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("LOW-YIELD DESTRUCTIVE MECHANISMS — MEKONG DELTA 2025\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. CONTRAST EVENTS (thường gặp ở Low, hiếm ở High)\n")
            f.write("-" * 50 + "\n")
            for r in contrast_report:
                f.write(f"• {r['event']}\n")
                f.write(
                    f"   Low: {r['support_low_%']}%\tHigh: {r['support_high_%']}%\tGrowth: {r['growth']}×\n\n"
                )

            f.write("2. RARE BUT CATASTROPHIC PATTERNS\n")
            f.write("-" * 50 + "\n")
            for r in rare_report:
                f.write(f"• {r['pattern']}\n")
                f.write(
                    f"   Freq in Low: {r['freq_in_low']} vụ → Yield drop: {r['yield_drop']*100:.1f}%\n\n"
                )

            f.write("3. GOLDEN SEQUENCE BREAKERS\n")
            f.write("-" * 50 + "\n")
            for r in breaker_report:
                f.write(f"• Golden: {r['golden_pattern']}\n")
                f.write(f"   → Phá bởi: {r['broken_by']} ({r['percentage']}% các trường hợp)\n\n")

        # 2. CSV + JSON
        pd.DataFrame(contrast_report).to_csv(Path(output_dir) / "contrast_events.csv", index=False)
        pd.DataFrame(rare_report).to_csv(Path(output_dir) / "rare_destructive.csv", index=False)
        pd.DataFrame(breaker_report).to_csv(Path(output_dir) / "golden_breakers.csv", index=False)

        # 3. Summary JSON (dễ tích hợp)
        summary = {
            "n_contrast_events": len(contrast_report),
            "n_rare_patterns": len(rare_report),
            "n_breakers": len(breaker_report),
            "top_killers": [r["event"] for r in contrast_report],
        }
        with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"LOW-YIELD ANALYSIS COMPLETE → saved to {output_dir}")
        logger.info(
            f"   Found {len(contrast_report)} contrast events | "
            f"{len(rare_report)} rare killers | {len(breaker_report)} breakers"
        )
