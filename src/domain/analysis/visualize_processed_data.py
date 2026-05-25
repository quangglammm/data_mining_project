"""Advanced mining & visualization of processed rice yield data (2025)."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import numpy as np
from collections import Counter
import ast

# === Config ===
EXPORT_DIR = Path("data/exports")
SEQ_FILE = EXPORT_DIR / "04_event_sequences.csv"
AGG_FILE = EXPORT_DIR / "03_aggregated_features.csv"


@st.cache_data
def load_data():
    seq_df = pd.read_csv(SEQ_FILE)
    agg_df = pd.read_csv(AGG_FILE)

    # Fix event_sequence column
    seq_df["event_sequence"] = seq_df["event_sequence"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else []
    )
    return seq_df, agg_df


seq_df, agg_df = load_data()
merged_df = seq_df.merge(agg_df, on=["id_vụ", "year", "yield_class"], how="left")

st.title("Rice Yield Prediction — Advanced Data Mining Dashboard (2025)")
st.sidebar.header("Controls")

# === 1. Sankey Diagram: Weather Journey by Yield Class ===
st.header("1. Weather Journey Flow (Sankey Diagram)")
yield_class = st.sidebar.selectbox("Yield Class", ["All", "High", "Medium", "Low"])


def build_sankey(df):
    if yield_class != "All":
        df = df[df["yield_class"] == yield_class]

    # Extract all transitions: event i → event i+1
    links = []
    for seq in df["event_sequence"]:
        for i in range(len(seq) - 1):
            links.append((seq[i], seq[i + 1]))

    link_counts = Counter(links)
    nodes = sorted(set([item for t in link_counts.keys() for item in t]))
    node_map = {node: i for i, node in enumerate(nodes)}

    source = [node_map[a] for (a, b) in link_counts.keys()]
    target = [node_map[b] for (a, b) in link_counts.keys()]
    value = list(link_counts.values())
    labels = [
        n.replace("_", " ")
        .replace("Vừa", "Moderate")
        .replace("Nóng", "Hot")
        .replace("Mát", "Cool")
        .replace("Ướt", "Wet")
        .replace("Khô", "Dry")
        for n in nodes
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(pad=15, thickness=20, label=labels, color="lightblue"),
                link=dict(source=source, target=target, value=value, color="rgba(100,150,255,0.4)"),
            )
        ]
    )
    fig.update_layout(title_text=f"Weather Event Flow → {yield_class} Yield", font_size=12)
    st.plotly_chart(fig, use_container_width=True)


build_sankey(seq_df if yield_class == "All" else seq_df[seq_df["yield_class"] == yield_class])

# === 2. Top Contrast Patterns (from your mining) ===
st.header("2. Top High vs Low Yield Weather Patterns")
contrast_file = Path("output/latest_run/contrast/contrast_patterns.csv")
if contrast_file.exists():
    contrast_df = pd.read_csv(contrast_file)
    contrast_df["events"] = contrast_df["events"].apply(ast.literal_eval)

    high = contrast_df[contrast_df["type"] == "High-yield marker"].head(10)
    low = contrast_df[contrast_df["type"] == "Low-yield marker"].head(10)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("High-Yield Markers")
        for _, row in high.iterrows():
            pattern = " → ".join([e.split("_")[1] for e in row["events"]])
            st.success(f"**{pattern}** → {row['growth_rate']:.1f}× more in High")
    with col2:
        st.subheader("Low-Yield Markers")
        for _, row in low.iterrows():
            pattern = " → ".join([e.split("_")[1] for e in row["events"]])
            st.error(f"**{pattern}** → {row['growth_rate']:.1f}× more in Low")
else:
    st.warning("Run `train` command first to generate contrast patterns!")

# === 3. UMAP of Sequences (Symbolic Embedding) ===
st.header("3. 2D Visualization of Weather Sequences (UMAP)")
if st.button("Run UMAP (may take 30s)"):
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # Convert sequences to strings
    seq_strings = [" ".join(seq) for seq in seq_df["event_sequence"]]
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w[\w-]+\w\b")
    X = vectorizer.fit_transform(seq_strings)

    umap_2d = UMAP(n_components=2, random_state=42)
    proj_2d = umap_2d.fit_transform(X.toarray())

    fig = px.scatter(
        x=proj_2d[:, 0],
        y=proj_2d[:, 1],
        color=seq_df["yield_class"],
        hover_data={"id_vụ": seq_df["id_vụ"], "year": seq_df["year"]},
        title="UMAP of Weather Event Sequences",
        labels={"color": "Yield Class"},
    )
    st.plotly_chart(fig, use_container_width=True)

# === 4. Climate Shift Over Time ===
st.header("4. Climate Change Impact: Pattern Frequency Over Years")
pattern = st.selectbox(
    "Select pattern to track",
    ["Flowering_Nóng-Khô", "Flowering_Nóng-Ướt", "Ripening_Mát-Vừa", "Tillering_Vừa-Ướt"],
)

yearly = []
for year in sorted(merged_df["year"].unique()):
    year_df = merged_df[merged_df["year"] == year]
    freq = sum(1 for seq in year_df["event_sequence"] if pattern in seq) / len(year_df)
    yearly.append({"year": year, "frequency": freq * 100})

freq_df = pd.DataFrame(yearly)
fig = px.line(freq_df, x="year", y="frequency", title=f"Frequency of '{pattern}' Over Time")
fig.update_yaxes(title="% of Seasons")
st.plotly_chart(fig, use_container_width=True)

# === 5. Numerical Feature Correlation Heatmap ===
st.header("5. Numerical Feature Importance by Yield Class")
numeric_cols = [c for c in agg_df.columns if c not in ["id_vụ", "year", "yield_class"]]
corr = agg_df[numeric_cols + ["yield_class"]].groupby("yield_class").mean().T
fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Average Feature Value by Yield Class",
)
st.plotly_chart(fig, use_container_width=True)

if st.button("Generate PDF Report"):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rice Yield Climate Pattern Report 2025", ln=1, align="C")
    # Add screenshots or text
    pdf.output("reports/climate_report_2025.pdf")
    st.success("PDF report generated!")
