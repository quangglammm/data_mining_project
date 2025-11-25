import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Cấu hình
CSV_FILE_PATH = "data/exports/03_aggregated_features_v2.csv"  # Sửa nếu cần path đầy đủ
OUTPUT_FOLDER = "weather_distribution"  # Thư mục lưu biểu đồ
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Định nghĩa giai đoạn (dựa trên code của bạn, stage_1 đến stage_5)
GROWTH_STAGES = {
    "stage_1": "mạ",
    "stage_2": "đẻ nhánh",
    "stage_3": "làm đòng",
    "stage_4": "trổ bông",
    "stage_5": "chín"
}

# Đọc file
print("Đang đọc file CSV...")
df = pd.read_csv(CSV_FILE_PATH)
print(f"Đã đọc {len(df)} mùa vụ")

# Trích xuất province từ id_vụ
df['province'] = df['id_vụ'].apply(lambda x: x.split('_')[0])

# Tạo DataFrame results cho phân bố
results = []
for idx, row in df.iterrows():
    for stage_num, stage_name in GROWTH_STAGES.items():
        avg_temp_col = f"{stage_num}_avg_temp"
        total_precip_col = f"{stage_num}_total_precip"
        
        if avg_temp_col in row and total_precip_col in row:
            results.append({
                "id_vụ": row["id_vụ"],
                "year": row["year"],
                "province": row["province"],
                "stage": stage_name,
                "avg_temp": row[avg_temp_col],
                "total_precip": row[total_precip_col]
            })

df_results = pd.DataFrame(results)
print(f"Hoàn thành! Có {len(df_results)} bản ghi giai đoạn để phân tích.\n")

# Hiển thị kết quả text (describe)
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 12

for stage in GROWTH_STAGES.values():
    stage_data = df_results[df_results["stage"] == stage]
    
    print(f"\n=== GIAI ĐOẠN: {stage.upper()} ===")
    print(stage_data[["avg_temp", "total_precip"]].describe())
    
    # Vẽ histogram: nhiệt độ và lượng mưa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(data=stage_data, x="avg_temp", kde=True, ax=ax1, color="orange", bins=30)
    ax1.set_title(f"Phân bố nhiệt độ trung bình - {stage}")
    ax1.set_xlabel("Nhiệt độ trung bình (°C)")
    ax1.axvline(stage_data["avg_temp"].quantile(0.33), color='r', linestyle='--', label='1/3')
    ax1.axvline(stage_data["avg_temp"].quantile(0.67), color='r', linestyle='--', label='2/3')
    ax1.legend()
    
    sns.histplot(data=stage_data, x="total_precip", kde=True, ax=ax2, color="blue", bins=30)
    ax2.set_title(f"Phân bố tổng lượng mưa - {stage}")
    ax2.set_xlabel("Tổng lượng mưa (mm)")
    ax2.axvline(stage_data["total_precip"].quantile(0.33), color='r', linestyle='--', label='1/3')
    ax2.axvline(stage_data["total_precip"].quantile(0.67), color='r', linestyle='--', label='2/3')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{stage}_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()  # Đóng figure để tránh hiển thị nếu chạy batch

# Biểu đồ boxplot theo tỉnh
print("\nĐang vẽ biểu đồ so sánh giữa các tỉnh...")
for stage in GROWTH_STAGES.values():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    sns.boxplot(data=df_results[df_results["stage"] == stage], 
                x="province", y="avg_temp", palette="Set2", ax=ax1)
    ax1.set_title(f"Nhiệt độ trung bình giai đoạn {stage} theo tỉnh")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    sns.boxplot(data=df_results[df_results["stage"] == stage], 
                x="province", y="total_precip", palette="Set3", ax=ax2)
    ax2.set_title(f"Tổng lượng mưa giai đoạn {stage} theo tỉnh")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/{stage}_by_province.png", dpi=200, bbox_inches='tight')
    plt.close()

print(f"\nHoàn tất! Tất cả biểu đồ đã được lưu vào thư mục: {OUTPUT_FOLDER}")
print("Mở thư mục để xem histogram (phân bố tổng) và boxplot (theo tỉnh).")
print("Dựa vào output text (describe), bạn có thể thấy mean, min, max, std, và quartiles để quyết định ngưỡng.")