# ================================
# generate_symbolic_sequences_from_aggregated.py
# Chạy độc lập – chỉ cần file 03_aggregated_features_v2.csv
# Kết quả: rice_event_sequences_fixed.csv (đúng format df_sequences)
# Thời gian chạy: ~5-10 giây cho 1001 mùa vụ
# ================================

import pandas as pd

# 1. NGƯỠNG CỐ ĐỊNH TỐI ƯU (copy từ trước)
FIXED_THRESHOLDS = {
    "mạ":        {"temp": {"Mát": (-99, 27.0),  "Vừa": (27.0, 28.5),  "Nóng": (28.5, 999)},
                  "precip": {"Khô": (-99, 70),   "Vừa": (70, 150),     "Ướt": (150, 9999)}},
    "đẻ nhánh":  {"temp": {"Mát": (-99, 26.5),  "Vừa": (26.5, 28.0),  "Nóng": (28.0, 999)},
                  "precip": {"Khô": (-99, 100),  "Vừa": (100, 200),    "Ướt": (200, 9999)}},
    "làm đòng":  {"temp": {"Mát": (-99, 26.5),  "Vừa": (26.5, 27.5),  "Nóng": (27.5, 999)},
                  "precip": {"Khô": (-99, 50),   "Vừa": (50, 150),     "Ướt": (150, 9999)}},
    "trổ bông":  {"temp": {"Mát": (-99, 26.2),  "Vừa": (26.2, 27.0),  "Nóng": (27.0, 999)},
                  "precip": {"Khô": (-99, 70),   "Vừa": (70, 190),     "Ướt": (190, 9999)}},
    "chín":      {"temp": {"Mát": (-99, 26.5),  "Vừa": (26.5, 27.5),  "Nóng": (27.5, 999)},
                  "precip": {"Khô": (-99, 80),   "Vừa": (80, 230),     "Ướt": (230, 9999)}}
}

# Mapping stage số → tên tiếng Việt
STAGE_MAPPING = {
    1: "mạ",
    2: "đẻ nhánh",
    3: "làm đòng",
    4: "trổ bông",
    5: "chín"
}

def get_label(value, thresholds):
    for label, (low, high) in thresholds.items():
        if low <= value < high:
            return label
    return list(thresholds.keys())[-1]

# 2. Đọc file đã có
print("Đang đọc data/exports/03_aggregated_features_v2.csv...")
df = pd.read_csv("data/exports/03_aggregated_features_v2.csv")
print(f"Đọc xong {len(df)} mùa vụ")

# 3. Tạo symbolic sequences
sequences = []

for _, row in df.iterrows():
    event_list = []
    
    for stage_num in range(1, 6):
        stage_name_vn = STAGE_MAPPING[stage_num]
        
        temp_col = f"stage_{stage_num}_avg_temp"
        precip_col = f"stage_{stage_num}_total_precip"
        
        if temp_col not in row or precip_col not in row:
            continue
            
        avg_temp = row[temp_col]
        total_precip = row[precip_col]
        
        t_label = get_label(avg_temp, FIXED_THRESHOLDS[stage_name_vn]["temp"])
        p_label = get_label(total_precip, FIXED_THRESHOLDS[stage_name_vn]["precip"])
        
        event_list.append(f"stage_{stage_num}_{t_label}-{p_label}")
    
    sequences.append({
        "id_vụ": row["id_vụ"],
        "year": row["year"],
        "yield_class": row["yield_class"],
        "event_sequence": event_list  # danh sách đúng như bạn muốn
    })

# 4. Xuất ra file
result_df = pd.DataFrame(sequences)
output_file = "data/exports/rice_event_sequences_fixed.csv"
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\nHOÀN TẤT trong vài giây!")
print(f"Đã tạo {len(result_df)} chuỗi symbolic cố định")
print(f"File lưu tại: {output_file}")
print("\nVí dụ 3 dòng đầu:")
print(result_df.head(3)[["id_vụ", "yield_class", "event_sequence"]].to_string(index=False))

# Bonus: xem chuỗi phổ biến nhất theo nhãn
print("\nTop 5 chuỗi phổ biến theo yield_class:")
print(result_df.explode("event_sequence")
      .groupby(["yield_class", "event_sequence"]).size()
      .groupby(level=0).nlargest(5))