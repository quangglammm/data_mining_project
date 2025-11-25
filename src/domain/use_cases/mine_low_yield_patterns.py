# src/domain/use_cases/mine_low_yield_patterns.py
from typing import List, Tuple, Dict, Any
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MineLowYieldPatternsUseCase:
    def __init__(
        self,
        min_support_low: float = 0.10,
        growth_threshold: float = 2.0,
        rare_min_count: int = 3,
        rare_max_count: int = 6,
        rare_min_low_ratio: float = 0.95,
        top_breaker_count: int = 10,  # ← THÊM THAM SỐ MỚI
    ):
        self.min_support_low = min_support_low
        self.growth_threshold = growth_threshold
        self.rare_min_count = rare_min_count
        self.rare_max_count = rare_max_count
        self.rare_min_low_ratio = rare_min_low_ratio
        self.top_breaker_count = top_breaker_count  # ← số lượng breaker lấy ra

    def execute(
        self,
        df_sequences: pd.DataFrame,
        high_golden_patterns: Any = None,
        output_dir: str = "output/latest_run/destructive",
    ) -> Dict[str, Any]:

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        high_mask = df_sequences["yield_class"] == "High"
        low_mask = df_sequences["yield_class"] == "Low"
        high_seqs = df_sequences[high_mask]["event_sequence"].apply(self._to_list).tolist()
        low_seqs = df_sequences[low_mask]["event_sequence"].apply(self._to_list).tolist()
        n_high, n_low = len(high_seqs), len(low_seqs)

        def full_event(e: str) -> str:
            return e if isinstance(e, str) and "_" in e else e

        high_full = [[full_event(e) for e in seq] for seq in high_seqs]
        low_full = [[full_event(e) for e in seq] for seq in low_seqs]

        contrast_events: List[Tuple[str, ...]] = []
        destructive_patterns: List[Tuple[str, ...]] = []
        breaker_events: List[Tuple[str, ...]] = []

        # =================================================================
        # 1. TỬ HUYỆT PHỔ BIẾN (giữ nguyên)
        # =================================================================
        cnt_low = Counter(e for seq in low_full for e in seq)
        cnt_high = Counter(e for seq in high_full for e in seq)

        contrast_report = []
        for event, c_low in cnt_low.items():
            supp_low = c_low / n_low
            supp_high = cnt_high[event] / n_high
            growth = supp_low / (supp_high + 1e-8)
            if supp_low >= self.min_support_low and growth >= self.growth_threshold:
                contrast_events.append((event,))
                contrast_report.append(
                    {
                        "events": (event,),
                        "pattern": event,
                        "support_low_%": round(supp_low * 100, 2),
                        "support_high_%": round(supp_high * 100, 2),
                        "growth": round(growth, 2),
                    }
                )

        # =================================================================
        # 2. SÁT THỦ THẦM LẶNG (giữ nguyên)
        # =================================================================
        bigram_cnt = Counter()
        bigram_yield = defaultdict(list)
        for _, row in df_sequences.iterrows():
            seq = [full_event(e) for e in self._to_list(row["event_sequence"])]
            yclass = row["yield_class"]
            for i in range(len(seq) - 1):
                pat = (seq[i], seq[i + 1])
                bigram_cnt[pat] += 1
                bigram_yield[pat].append(yclass)

        rare_report = []
        for (e1, e2), total_cnt in bigram_cnt.items():
            if not (self.rare_min_count <= total_cnt <= self.rare_max_count):
                continue
            low_ratio = bigram_yield[(e1, e2)].count("Low") / total_cnt
            if low_ratio >= self.rare_min_low_ratio:
                pattern_tuple = (e1, e2)
                destructive_patterns.append(pattern_tuple)
                rare_report.append(
                    {
                        "events": pattern_tuple,
                        "pattern": f"{e1} to {e2}",
                        "count": total_cnt,
                        "low_ratio_%": round(low_ratio * 100, 1),
                    }
                )

        # =================================================================
        # 3. ĐIỂM GÃY CHUỖI VÀNG – ĐÃ ĐƯỢC CẬP NHẬT: LẤY TẦN SUẤT CAO NHẤT
        # =================================================================
        high_tuples = [tuple(seq) for seq in high_full]
        top_golden = Counter(high_tuples).most_common(5)

        all_breakers = []
        for golden_tuple, _ in top_golden:
            for low_seq in low_full:
                if len(low_seq) < 5:
                    continue
                diffs = [i for i, (a, b) in enumerate(zip(low_seq, golden_tuple)) if a != b]
                if len(diffs) == 1:
                    bad_event = low_seq[diffs[0]]
                    all_breakers.append(bad_event)

        # ĐÃ SỬA: LẤY CÁC EVENT XUẤT HIỆN NHIỀU NHẤT (tác động mạnh nhất)
        breaker_counter = Counter(all_breakers)
        top_breakers_with_count = breaker_counter.most_common(self.top_breaker_count)

        breaker_report = []
        for event, count in top_breakers_with_count:
            if count < 2:  # có thể lọc thêm nếu muốn
                continue
            tup = (event,)
            breaker_events.append(tup)
            breaker_report.append(
                {
                    "events": tup,
                    "pattern": event,
                    "break_count": count,
                    "percentage": round(count / len(all_breakers) * 100, 1) if all_breakers else 0,
                    "note": f"Phá vỡ chuỗi vàng {count} lần",
                }
            )

        # =================================================================
        # TRẢ VỀ KẾT QUẢ
        # =================================================================
        self._save_results(
            output_dir=output_dir,
            contrast_report=contrast_report,
            rare_report=rare_report,
            breaker_report=breaker_report,
        )

        all_low_yield_pattern_tuples = contrast_events + destructive_patterns + breaker_events

        return {
            "contrast_events": contrast_events,
            "destructive_patterns": destructive_patterns,
            "breaker_events": breaker_events,
            "all_low_yield_pattern_tuples": all_low_yield_pattern_tuples,
            "reports": {
                "1_common_killer": contrast_report,
                "2_rare_deadly": rare_report,
                "3_golden_path_breaks": breaker_report,
            },
        }

    # =================================================================
    # Helper (giữ nguyên)
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

    def _save_results(self, output_dir: str, contrast_report, rare_report, breaker_report):
        txt_path = Path(output_dir) / "LOW_YIELD_3_GROUPS_TUPLE_STANDARD.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("3 NHÓM NGUY CƠ MẤT MÙA LÚA – PHIÊN BẢN CẬP NHẬT\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. TỬ HUYỆT PHỔ BIẾN\n")
            for r in contrast_report:
                f.write(f"• {r['pattern']} (growth {r['growth']:.1f}×)\n")

            f.write("\n2. SÁT THỦ THẦM LẶNG\n")
            for r in rare_report:
                f.write(f"• {r['pattern']} ({r['count']} lần → {r['low_ratio_%']}% Low)\n")

            f.write("\n3. ĐIỂM GÃY CHUỖI VÀNG – TẦN SUẤT CAO NHẤT\n")
            for r in breaker_report:
                f.write(
                    f"• {r['pattern']} → phá vỡ {r['break_count']} lần ({r['percentage']}% trường hợp)\n"
                )

        pd.DataFrame(contrast_report).to_csv(Path(output_dir) / "1_common_killer.csv", index=False)
        pd.DataFrame(rare_report).to_csv(Path(output_dir) / "2_rare_deadly.csv", index=False)
        pd.DataFrame(breaker_report).to_csv(
            Path(output_dir) / "3_golden_path_breaks.csv", index=False
        )

        logger.info(f"HOÀN TẤT 3 NHÓM PATTERN → {output_dir}")
