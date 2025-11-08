import pandas as pd
import numpy as np
from datetime import timedelta
from prefixspan import PrefixSpan

# --- Thư viện mới cho Model Building ---
# Cần cài đặt: pip install scikit-learn xgboost matplotlib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings

# Tắt các cảnh báo (ví dụ: từ XGBoost)
warnings.filterwarnings('ignore')

# === CÁC BIẾN CẦN BẠN CUNG CẤP ===
RICE_DATA_FILE = "data/DBSCL_agriculture_1995_2024.csv"
WEATHER_DATA_FILE = "data/DBSCL_weather_1995_2024_FULL.xlsx"

"""
 - winter - spring: from Novemver/December last year to March/April this year
 - summer - autumn: from April/May to August/September
 - main season    : from May/June to late November
"""
SEASON_DEFINITIONS = {
    'winter_spring': {'start_month': 11, 'start_day': 15, 'end_month': 3, 'end_day': 15, 'year_offset': -1},
    'summer_autumn': {'start_month': 4, 'start_day': 15, 'end_month': 8, 'end_day': 15, 'year_offset': 0},
    'main_season': {'start_month': 5, 'start_day': 15, 'end_month': 11, 'end_day': 30, 'year_offset': 0}
}

"""
Five growth stages:
 - stage_1: Mạ
 - stage_2: Đẻ nhánh
 - stage_3: Làm đòng
 - stage_4: Trổ
 - stage_5: Chín
"""
GROWTH_STAGE_DEFINITIONS = {
    'stage_1': (0, 20),
    'stage_2': (21, 45),
    'stage_3': (46, 60),
    'stage_4': (61, 80),
    'stage_5': (81, 105)
}
# === KẾT THÚC PHẦN CÀI ĐẶT ===


def robust_qcut(x, q=3, labels=None):
    """
    Hàm qcut mạnh mẽ (FIX LỖI), sử dụng 'rank' để xử lý
    các giá trị trùng lặp mà 'duplicates=drop' không xử lý được.
    """
    if labels is None:
        labels = ['Low', 'Medium', 'High']

    try:
        # Thử qcut bình thường trước
        return pd.qcut(x, q=q, labels=labels, duplicates='drop')
    except ValueError:
        # Nếu qcut bình thường thất bại do bin không duy nhất:
        return pd.qcut(x.rank(method='first'), q=q, labels=labels, duplicates='drop')


def process_rice_data(input_file):
    """
    Thực hiện Bước 1, 2, 3: Melt, Filter, và Discretize dữ liệu lúa.
    """
    print("--- (Bước 1-3) Đang xử lý dữ liệu lúa... ---")
    try:
        df_agri = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp {input_file}. Vui lòng kiểm tra lại.")
        return None

    # 1. Melt (Wide-to-Long)
    df_long = pd.wide_to_long(df_agri,
                              stubnames=['cultivated_area', 'rice_yield', 'rice_production'],
                              i=['province', 'year'],
                              j='season',
                              sep='_',
                              suffix='.*').reset_index()

    # 2. Filter (Lọc)
    df_filtered = df_long[df_long['cultivated_area'] > 1.0].copy()

    # 3. Discretize (Rời rạc hóa, nhóm theo Tỉnh và Vụ)
    labels = ['Low', 'Medium', 'High']

    # Gán nhãn lớp (FIX: Sử dụng hàm robust_qcut)
    df_filtered['yield_class'] = df_filtered.groupby(['province', 'season'])['rice_yield'].transform(
        lambda x: robust_qcut(x, q=3, labels=labels)
    )

    # Bỏ các hàng không thể gán nhãn (do không đủ dữ liệu trong nhóm)
    df_filtered = df_filtered.dropna(subset=['yield_class'])

    print("--- (Bước 1-3) Xử lý dữ liệu lúa hoàn tất. ---")
    return df_filtered


def load_weather_data(weather_file):
    """
    Tải và tiền xử lý cơ bản dữ liệu thời tiết.
    """
    print(f"--- Đang tải dữ liệu thời tiết từ {weather_file}... ---")
    try:
        df_weather = pd.read_excel(weather_file)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp {weather_file}. Vui lòng tải lên và đặt tên đúng.")
        return None
    except ImportError:
        print("LỖI: Thiếu thư viện 'openpyxl'. Vui lòng chạy 'pip install openpyxl' để đọc tệp Excel.")
        return None

    # Đảm bảo cột 'date' là kiểu datetime
    df_weather['date'] = pd.to_datetime(df_weather['date'])

    # Xử lý dữ liệu thiếu (ví dụ: nội suy)
    df_weather = df_weather.sort_values(by=['province', 'date'])
    df_weather = df_weather.set_index('date') # Bắt buộc cho interpolate time

    numeric_cols = ['max_temp', 'min_temp', 'mean_temp', 'precipitation_sum', 'humidity_mean', 'et0_mm']
    for col in numeric_cols:
        if col in df_weather.columns:
            df_weather[col] = df_weather.groupby('province')[col].transform(lambda x: x.interpolate(method='time'))

    df_weather = df_weather.reset_index() # Trả 'date' về làm cột

    if 'max_temp' in df_weather.columns and 'min_temp' in df_weather.columns:
        df_weather['dtr'] = df_weather['max_temp'] - df_weather['min_temp']

    print("--- Tải và xử lý dữ liệu thời tiết hoàn tất. ---")
    return df_weather


def align_weather_data(df_rice, df_weather, season_definitions):
    """
    Bước 4: Map chuỗi thời tiết hàng ngày vào từng mùa vụ.
    """
    print("--- (Bước 4) Đang thực hiện Data Alignment... ---")
    if df_weather is None: return None

    weather_sequences = []
    df_weather_indexed = df_weather.set_index(['province', 'date'])

    for _, row in df_rice.iterrows():
        province = row['province']
        year = row['year']
        season = row['season']

        try:
            season_def = season_definitions[season]
        except KeyError:
            continue

        year_offset = season_def['year_offset']
        start_date = pd.to_datetime(f"{year + year_offset}-{season_def['start_month']}-{season_def['start_day']}")
        end_date = pd.to_datetime(f"{year}-{season_def['end_month']}-{season_def['end_day']}")

        try:
            daily_weather = df_weather_indexed.loc[
                (province, start_date) : (province, end_date)
            ].reset_index()

            if not daily_weather.empty:
                weather_sequences.append({
                    'id_vụ': f"{province}_{year}_{season}",
                    'year': year, # Thêm cột year để sắp xếp
                    'yield_class': row['yield_class'],
                    'daily_weather_sequence': daily_weather
                })
        except KeyError:
            pass

    print(f"--- (Bước 4) Data Alignment hoàn tất. Tìm thấy {len(weather_sequences)} chuỗi mùa vụ. ---")
    return pd.DataFrame(weather_sequences)


def create_sequence_database(df_aligned, stage_definitions):
    """
    Bước 5: Tạo Cơ sở dữ liệu Chuỗi Sự kiện (Event Sequence Database).
    Trả về: df_agg (đặc trưng số), df_sequences (đặc trưng chuỗi)
    """
    print("--- (Bước 5) Đang tạo Cơ sở dữ liệu Chuỗi Sự kiện... ---")

    aggregated_stages = [] # Nơi lưu trữ dữ liệu tổng hợp (cho Đặc trưng số)
    event_sequences = []   # Nơi lưu trữ chuỗi sự kiện (cho Đặc trưng Mẫu)

    # --- 5a. Tính toán ngưỡng rời rạc hóa (Discretization Thresholds) ---
    # Để tính ngưỡng, chúng ta cần duyệt qua 1 lần để lấy toàn bộ dữ liệu tổng hợp
    print("... (Bước 5a) Đang tính toán ngưỡng rời rạc...")
    temp_agg_data = {stage: [] for stage in stage_definitions.keys()}
    precip_agg_data = {stage: [] for stage in stage_definitions.keys()}

    for _, row in df_aligned.iterrows():
        daily_seq = row['daily_weather_sequence']
        if daily_seq.empty: continue
        start_date = daily_seq['date'].min()

        for stage_name, (start_day_rel, end_day_rel) in stage_definitions.items():
            stage_start_date = start_date + timedelta(days=start_day_rel)
            stage_end_date = start_date + timedelta(days=end_day_rel)
            stage_weather = daily_seq[
                (daily_seq['date'] >= stage_start_date) & (daily_seq['date'] <= stage_end_date)
            ]
            if stage_weather.empty: continue

            temp_agg_data[stage_name].append(stage_weather['mean_temp'].mean())
            precip_agg_data[stage_name].append(stage_weather['precipitation_sum'].sum())

    # Tính ngưỡng (quantiles) cho từng giai đoạn
    temp_thresholds = {stage: (pd.Series(data).quantile(1/3), pd.Series(data).quantile(2/3))
                       for stage, data in temp_agg_data.items() if data}
    precip_thresholds = {stage: (pd.Series(data).quantile(1/3), pd.Series(data).quantile(2/3))
                         for stage, data in precip_agg_data.items() if data}

    def get_event_label(temp, precip, t_thresh, p_thresh):
        if pd.isna(temp) or pd.isna(precip): return None
        if pd.isna(t_thresh[0]) or pd.isna(p_thresh[0]): return None # Không đủ data để tính ngưỡng

        if temp <= t_thresh[0]: t_label = "Mát"
        elif temp <= t_thresh[1]: t_label = "Vừa"
        else: t_label = "Nóng"

        if precip <= p_thresh[0]: p_label = "Khô"
        elif precip <= p_thresh[1]: p_label = "Vừa"
        else: p_label = "Ướt"

        return f"{t_label}-{p_label}" # Ví dụ: "Nóng-Khô"

    # --- 5b. Duyệt lần 2 để tạo CSDL cuối cùng ---
    print("... (Bước 5b) Đang tạo CSDL cuối cùng...")
    for _, row in df_aligned.iterrows():
        daily_seq = row['daily_weather_sequence']
        if daily_seq.empty: continue

        start_date = daily_seq['date'].min()

        stages_for_this_season = {} # Cho df_agg
        sequence_for_this_season = [] # Cho df_sequences

        stages_for_this_season['id_vụ'] = row['id_vụ']
        stages_for_this_season['year'] = row['year'] # Giữ lại 'year' để sắp xếp
        stages_for_this_season['yield_class'] = row['yield_class']

        for stage_name, (start_day_rel, end_day_rel) in stage_definitions.items():
            stage_start_date = start_date + timedelta(days=start_day_rel)
            stage_end_date = start_date + timedelta(days=end_day_rel)
            stage_weather = daily_seq[
                (daily_seq['date'] >= stage_start_date) & (daily_seq['date'] <= stage_end_date)
            ]

            if stage_weather.empty: continue

            # Tính đặc trưng số (cho df_agg)
            avg_temp = stage_weather['mean_temp'].mean()
            total_precip = stage_weather['precipitation_sum'].sum()

            stages_for_this_season[f'{stage_name}_avg_temp'] = avg_temp
            stages_for_this_season[f'{stage_name}_total_precip'] = total_precip
            stages_for_this_season[f'{stage_name}_count_heat_days'] = (stage_weather['max_temp'] > 35).sum()
            stages_for_this_season[f'{stage_name}_avg_et0'] = stage_weather['et0_mm'].mean()

            # Tạo sự kiện chuỗi (cho df_sequences)
            event = get_event_label(avg_temp, total_precip,
                                    temp_thresholds.get(stage_name, (0,0)),
                                    precip_thresholds.get(stage_name, (0,0)))

            if event:
                sequence_for_this_season.append(f"{stage_name}_{event}")

        aggregated_stages.append(stages_for_this_season)
        if sequence_for_this_season:
            event_sequences.append({
                'id_vụ': row['id_vụ'],
                'year': row['year'], # Giữ lại 'year' để sắp xếp
                'yield_class': row['yield_class'],
                'event_sequence': sequence_for_this_season
            })

    if not aggregated_stages or not event_sequences:
        print("LỖI (Bước 5): Không có dữ liệu sau khi tổng hợp.")
        return None, None

    df_agg = pd.DataFrame(aggregated_stages)
    df_sequences = pd.DataFrame(event_sequences)

    print(f"--- (Bước 5) Tạo CSDL chuỗi hoàn tất. {len(df_sequences)} chuỗi sự kiện được tạo. ---")
    # Trả về CẢ HAI dataframe
    return df_agg, df_sequences


def mine_sequential_patterns(df_sequences, min_support=0.1, minlen=2, maxlen=5):
    """
    Bước 6: Khai phá Mẫu Tuần tự (dùng PrefixSpan).
    Trả về: (set) của tất cả các mẫu duy nhất tìm được
    """
    print("--- (Bước 6) Đang khai phá Mẫu Tuần tự... ---")

    if df_sequences is None or df_sequences.empty:
        print("LỖI (Bước 6): Không có CSDL chuỗi để khai phá.")
        return set() # Trả về set rỗng

    # Chuyển đổi CSDL sang định dạng list-of-lists
    db_low = df_sequences[df_sequences['yield_class'] == 'Low']['event_sequence'].tolist()
    db_medium = df_sequences[df_sequences['yield_class'] == 'Medium']['event_sequence'].tolist()
    db_high = df_sequences[df_sequences['yield_class'] == 'High']['event_sequence'].tolist()

    print(f"Tổng số chuỗi: Low={len(db_low)}, Medium={len(db_medium)}, High={len(db_high)}")

    master_pattern_set = set() # Nơi lưu trữ tất cả các mẫu duy nhất

    # Hàm trợ giúp để chạy và lọc
    def find_and_filter(db, support_percent, name):
        if not db:
            print(f"Không có dữ liệu cho lớp '{name}', bỏ qua.")
            return []

        support_count = int(len(db) * support_percent)
        print(f"\n--- Đang tìm mẫu cho lớp '{name}' (min_support = {support_percent * 100}%, count >= {support_count}) ---")

        ps = PrefixSpan(db)
        patterns_gen = ps.frequent(support_count)

        filtered_patterns = [
            tuple(pat) for freq, pat in patterns_gen # Chuyển list (pat) thành tuple
            if minlen <= len(pat) <= maxlen
        ]

        # Sắp xếp và lấy top 5 để in
        top_5 = sorted(filtered_patterns, key=len, reverse=True)[:5]
        print(f"Top 5 mẫu tìm được cho '{name}': {top_5}")

        return filtered_patterns

    # Chạy khai phá trên cả 3 lớp
    patterns_low = find_and_filter(db_low, min_support, 'Low')
    patterns_medium = find_and_filter(db_medium, min_support, 'Medium')
    patterns_high = find_and_filter(db_high, min_support, 'High')

    # Tập hợp tất cả các mẫu duy nhất
    master_pattern_set.update(patterns_low)
    master_pattern_set.update(patterns_medium)
    master_pattern_set.update(patterns_high)

    print(f"\n--- (Bước 6) Khai phá mẫu hoàn tất. Tìm thấy {len(master_pattern_set)} mẫu duy nhất. ---")
    return master_pattern_set


def build_feature_matrix(df_agg, df_sequences, all_patterns):
    """
    Bước 7: Xây dựng Ma trận Đặc trưng (X) và Vector Mục tiêu (y)
    """
    print("--- (Bước 7) Đang xây dựng Ma trận Đặc trưng (Hybrid)... ---")

    # --- Xây dựng Đặc trưng Mẫu (Pattern Features) ---
    print(f"... Đang tạo {len(all_patterns)} đặc trưng Mẫu (Pattern)...")
    # Chuyển chuỗi sự kiện thành 'set' để tìm kiếm nhanh hơn
    sequences_as_sets = df_sequences['event_sequence'].apply(set).tolist()

    pattern_features = []

    for pattern in all_patterns:
        pattern_set = set(pattern)
        col_name = f"pat_{'__'.join(pattern)}" # Tạo tên cột duy nhất
        # Kiểm tra xem 'pattern_set' có phải là tập con của 'seq_set' không
        feature_col = [1 if pattern_set.issubset(seq_set) else 0 for seq_set in sequences_as_sets]
        pattern_features.append(pd.Series(feature_col, name=col_name))

    df_patterns = pd.concat(pattern_features, axis=1)

    # --- Xây dựng Đặc trưng Số (Numeric Features) ---
    print("... Đang tạo đặc trưng Số (Numeric)...")
    df_numeric = df_agg.drop(columns=['id_vụ', 'year', 'yield_class'])
    # Xử lý NaN (nếu có giai đoạn nào đó bị thiếu)
    df_numeric = df_numeric.fillna(0)

    # --- Kết hợp X và Y ---
    # Đảm bảo index của df_numeric và df_patterns khớp nhau
    df_numeric.index = df_sequences.index
    df_patterns.index = df_sequences.index

    # Nối đặc trưng số và đặc trưng pattern
    X = pd.concat([df_numeric, df_patterns], axis=1)

    # Lấy Target (y)
    df_target = df_sequences[['id_vụ', 'year', 'yield_class']]

    # --- Sắp xếp theo Thời gian (BẮT BUỘC) ---
    # Nối X và Y, sắp xếp theo 'year', sau đó tách ra lại
    df_final = pd.concat([X, df_target], axis=1)
    df_final = df_final.sort_values(by='year')
    df_final = df_final.reset_index(drop=True)

    # Tách X và Y đã được sắp xếp
    y_labels = df_final['yield_class']
    X_sorted = df_final.drop(columns=['id_vụ', 'yield_class'])

    # --- Mã hóa Target (Y) ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    print(f"--- (Bước 7) Xây dựng Đặc trưng hoàn tất. Kích thước X: {X_sorted.shape} ---")

    return X_sorted, y_encoded, X_sorted.columns.tolist(), le.classes_


def train_and_evaluate_model(X, y, feature_names, class_labels):
    """
    Bước 8 & 9: Phân chia, Huấn luyện và Đánh giá Mô hình
    """
    print("--- (Bước 8 & 9) Đang Huấn luyện và Đánh giá Mô hình... ---")

    # --- Bước 8: Phân chia (Time Series Split) ---
    tscv = TimeSeriesSplit(n_splits=5) # 5 Folds

    f1_scores = []
    acc_scores = []

    # Lặp qua các Folds
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Lấy năm min/max từ chính cột 'year'
        train_years = X_train['year']
        test_years = X_test['year']

        print(f"Training on {len(X_train)} samples (Years ~{train_years.min()} - ~{train_years.max()})")
        print(f"Testing on {len(X_test)} samples (Years ~{test_years.min()} - ~{test_years.max()})")

        # --- Bước 9: Huấn luyện (XGBoost) ---
        model = xgb.XGBClassifier(
            objective='multi:softmax', # Bài toán phân loại đa lớp
            num_class=len(class_labels),
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )

        model.fit(X_train, y_train)

        # Đánh giá
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        f1_scores.append(f1)
        acc_scores.append(acc)

        print(f"Fold {fold+1} F1-Score (macro): {f1:.4f}")
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    # --- Báo cáo Kết quả Tổng thể ---
    print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ (trên 5 Folds) ---")
    print(f"Average F1-Score (macro): {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    print(f"Average Accuracy: {np.mean(acc_scores):.4f} +/- {np.std(acc_scores):.4f}")

    # --- Phân tích Mô hình Cuối cùng (từ Fold cuối) ---
    print("\n--- Phân tích Mô hình Cuối cùng (Fold 5) ---")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_labels, ax=ax, cmap='Blues')
        plt.title(f"Confusion Matrix (Fold 5)")
        plt.savefig("confusion_matrix.png")
        print("Ma trận nhầm lẫn đã được lưu vào 'confusion_matrix.png'")
        plt.close(fig)
    except Exception as e:
        print(f"Lỗi khi vẽ Ma trận Nhầm lẫn: {e}")

    # --- Phân tích Đặc trưng Quan trọng ---
    try:
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        print("\n--- Top 15 Đặc trưng Quan trọng Nhất (từ Fold 5) ---")
        print(feat_imp.head(15))
    except Exception as e:
        print(f"Lỗi khi tính toán Đặc trưng Quan trọng: {e}")


# === HÀM CHẠY CHÍNH (MAIN) ===
def main():
    # Bước 1-3: Xử lý dữ liệu lúa
    df_rice_processed = process_rice_data(input_file=RICE_DATA_FILE)
    if df_rice_processed is None: return

    # Tải dữ liệu thời tiết
    df_weather = load_weather_data(WEATHER_DATA_FILE)
    if df_weather is None: return

    # Bước 4: Data Alignment
    df_aligned = align_weather_data(df_rice_processed, df_weather, SEASON_DEFINITIONS)
    if df_aligned is None: return

    # Bước 5: Tạo CSDL Chuỗi Sự kiện
    df_agg, df_sequences = create_sequence_database(df_aligned, GROWTH_STAGE_DEFINITIONS)
    if df_agg is None or df_sequences is None: return

    # Bước 6: Khai phá Mẫu
    master_pattern_list = mine_sequential_patterns(df_sequences, min_support=0.1, minlen=2, maxlen=4)

    # Bước 7: Xây dựng Ma trận Đặc trưng (X) và Vector Mục tiêu (y)
    X, y, feature_names, class_labels = build_feature_matrix(df_agg, df_sequences, master_pattern_list)

    # Bước 8 & 9: Huấn luyện và Đánh giá
    train_and_evaluate_model(X, y, feature_names, class_labels)


if __name__ == "__main__":
    main()

