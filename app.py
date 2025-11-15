import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import io

# ===========================
# Page Setup
# ===========================
st.set_page_config(page_title="BNA Histogram (Fixed Frequency)", layout="wide")
st.title("BNA Contour Histogram – Correct Frequency")
st.markdown("""
**Upload `.bna` files** → **X-axis = Parameter Value** (e.g., `0.15`, `200.0`)  
**Y-axis = Frequency = Number of Contour Lines** (each `"C",…` = 1 count)  
Fully customizable, high-res, matches your original plots.
""")

# ===========================
# File Uploader
# ===========================
uploaded_files = st.file_uploader(
    "Upload BNA Files",
    type=["bna"],
    accept_multiple_files=True,
    help="Porosity_Contours.bna, Permeability_Contours.bna, etc."
)

if not uploaded_files:
    st.info("Please upload at least one `.bna` file.")
    st.stop()

# ===========================
# BNA Parser – Count Each Contour Once
# ===========================
def parse_bna_contour_values(file_obj):
    """
    Reads .bna file and returns:
    - list of parameter values (second field in "C",...)
    - filename
    Ignores coordinates – we only need the contour label.
    """
    content = file_obj.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    values = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('"C"'):
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 3:
                label = parts[1]  # e.g., "0.150"
                try:
                    val = float(label)
                    values.append(val)
                except ValueError:
                    continue  # skip non-numeric labels
    return values

# ===========================
# Collect All Values + Source File
# ===========================
all_values = []
value_file_pairs = []  # (value, filename)

for up_file in uploaded_files:
    try:
        vals = parse_bna_contour_values(up_file)
        for v in vals:
            all_values.append(v)
            value_file_pairs.append((v, up_file.name))
    except Exception as e:
        st.error(f"Error reading **{up_file.name}**: {e}")

if not all_values:
    st.error("No valid numeric contour values found.")
    st.stop()

data = np.array(all_values)

# ===========================
# Sidebar Controls
# ===========================
with st.sidebar:
    st.header("Histogram Settings")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Contours", len(data))
        st.metric("Min", f"{data.min():.6g}")
    with col2:
        st.metric("Max", f"{data.max():.6g}")
        st.metric("Mean", f"{data.mean():.6g}")

    st.markdown("---")

    num_bins = st.slider("Number of Bins", 5, 200, 50, 5)
    chart_title = st.text_input("Chart Title", "Histogram of Contour Values")
    x_label = st.text_input("X-Axis Label", "Parameter Value")
    y_label = st.text_input("Y-Axis Label", "Frequency (Number of Contours)")

    col1, col2 = st.columns(2)
    with col1:
        x_min = st.number_input("X Min", value=float(data.min()), format="%.6g")
        y_min = st.number_input("Y Min", value=0.0)
    with col2:
        x_max = st.number_input("X Max", value=float(data.max()), format="%.6g")
        y_auto = st.checkbox("Y Max Auto", value=True)
        y_max = st.number_input("Y Max", value=float(int(data.max()) + 1), disabled=y_auto)

# ===========================
# Plot Histogram – Correct Frequency
# ===========================
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# Use 'bins' as edges → frequency = count per bin
counts, bin_edges = np.histogram(data, bins=num_bins, range=(x_min, x_max))
ax.bar(
    bin_edges[:-1],
    counts,
    width=np.diff(bin_edges),
    color="#4C72B0",
    edgecolor="black",
    alpha=0.85,
    linewidth=0.8
)

# Styling
ax.set_title(chart_title, fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel(x_label, fontsize=14)
ax.set_ylabel(y_label, fontsize=14)
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
ax.set_axisbelow(True)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, None if y_auto else y_max)

# Add frequency labels on top of bars
for i, count in enumerate(counts):
    if count > 0:
        x_pos = bin_edges[i] + (bin_edges[i+1] - bin_edges[i]) / 2
        ax.annotate(
            f"{int(count)}",
            xy=(x_pos, count),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black"
        )

plt.tight_layout()
st.pyplot(fig)

# ===========================
# Data Table
# ===========================
with st.expander("Raw Contour Values (One Row = One Contour Line)"):
    df = pd.DataFrame(value_file_pairs, columns=["Value", "Source File"])
    df = df.sort_values("Value").reset_index(drop=True)
    st.dataframe(df, use_container_width=True)
    st.caption(f"Total: {len(df)} contour lines")

# ===========================
# Download PNG
# ===========================
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
buf.seek(0)

st.download_button(
    label="Download Plot as PNG",
    data=buf,
    file_name="contour_histogram_correct_frequency.png",
    mime="image/png"
)

plt.close(fig)

# ===========================
# Footer
# ===========================
st.caption("Fixed: Frequency = Number of Contour Lines | X-axis = Parameter Label | Streamlit + Matplotlib")
