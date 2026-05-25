import pandas as pd
import matplotlib.pyplot as plt

# ------------------- SETTINGS -------------------
csv_file = "data/exports/01_labeled_yield_v2.csv"   # change if needed
province = "Long An"                   # change to any province you want
season   = "winter_spring"             # winter_spring / summer_autumn / main_season
# ------------------------------------------------

# Load data
df = pd.read_csv(csv_file)

# Filter
data = df[(df['province'] == province) & (df['season'] == season)].sort_values('year')

if data.empty:
    raise ValueError(f"No data found for {province} – {season}")

years        = data['year'].values
original     = data['rice_yield'].values          # actual yield
trend        = data['expected_yield'].values       # linear model prediction

# Plot
fig, ax1 = plt.subplots(figsize=(11, 6))

# Original and trend on left axis
ax1.plot(years, original, 'g-o', label='Original Yield', markersize=4)
ax1.plot(years, trend,   'b-',  linewidth=3, label='Linear Trend (expected_yield)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Rice Yield (ta/ha)', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Title
plt.title(f'Rice Yield Detrending – {province} ({season.replace("_", " ").title()})', fontsize=14, pad=15)

# Optional: add legend for residuals
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='upper left')

plt.tight_layout()
plt.savefig(f'detrend_{province}_{season}.png', dpi=300, bbox_inches='tight')
plt.show()