import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import io

# ===========================
# Page config
# ===========================
st.set_page_config(page_title="BNA Contour Histogram Analyzer", layout="wide")
st.title("BNA Contour Histogram Analyzer")
st.markdown(
    """
Upload one or more **.bna** contour files and generate fully-customizable histograms.
"""
)

# ===========================
# File uploader
# ===========================
uploaded_files = st.file_uploader(
    "Upload BNA Files",
    type=["bna"],
    accept_multiple_files=True,
    help="Select one or more .bna files"
)

if not uploaded_files:
    st.info("Please upload at least one .bna file to start.")
    st.stop()

# ===========================
# BNA parser
# ===========================
def parse_bna(file_obj):
    """Return a dict: contour_value → list of (x, y) points."""
    content = file_obj.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    contours = {}
    current_label = None
    current_points = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # New contour line: "C","label",count
        if line.startswith('"C"'):
            # Save previous contour
            if current_label is not None and current_points:
                contours[current_label] = current_points.copy()
            # Parse new header
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 3:
                label = parts[1]
                try:
                    value = float(parts[2])
                except ValueError:
                    value = label          # fallback
                current_label = value
                current_points = []
            continue

        # Coordinate line
        coords = re.split(r'[,;\s]+', line)
        if len(coords) >= 2:
            try:
                x, y = float(coords[0]), float(coords[1])
                current_points.append((x, y))
            except ValueError:
                continue

    # Save the last contour
    if current_label is not None and current_points:
        contours[current_label] = current_points

    return contours

# ===========================
# Parse all files & collect (value, filename) pairs
# ===========================
value_file_pairs = []      # list of (value, filename)
all_values = []            # just the numeric values for the histogram

for up_file in uploaded_files:
    try:
        contours = parse_bna(up_file)
        for raw_key in contours.keys():
            # Accept int/float or a string that looks like a number
            if isinstance(raw_key, (int, float)):
                val = float(raw_key)
            else:
                clean = raw_key.replace(".", "", 1).lstrip("-")
                if clean.isdigit() or (clean.replace("e", "", 1).replace("E", "", 1).isdigit()):
                    val = float(raw_key)
                else:
                    continue          # skip non-numeric labels
            value_file_pairs.append((val, up_file.name))
            all_values.append(val)
    except Exception as e:
        st.error(f"Error reading **{up_file.name}**: {e}")

if not all_values:
    st.error("No numeric contour values found in the uploaded files.")
    st.stop()

data = np.array(all_values)

# ===========================
# Sidebar – controls
# ===========================
with st.sidebar:
    st.header("Histogram Settings")

    st.metric("Total Values", len(data))
    st.metric("Min", f"{data.min():.6g}")
    st.metric("Max", f"{data.max():.6g}")
    st.metric("Mean", f"{data.mean():.6g}")

    st.markdown("---")

    num_bins = st.slider("Number of Bins", 5, 200, 50, 5)

    chart_title = st.text_input("Chart Title", "Histogram of Contour Values")
    x_label = st.text_input("X-Axis Label", "Contour Value")
    y_label = st.text_input("Y-Axis Label", "Frequency")

    col1, col2 = st.columns(2)
    with col1:
        x_min = st.number_input("X Min", value=float(data.min()), format="%.6g")
        y_min = st.number_input("Y Min", value=0.0)
    with col2:
        x_max = st.number_input("X Max", value=float(data.max()), format="%.6g")
        y_auto = st.checkbox("Y Max Auto", value=True)
        y_max = st.number_input("Y Max", value=float(data.max()), disabled=y_auto)

# ===========================
# Plot
# ===========================
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

n, bins, patches = ax.hist(
    data,
    bins=num_bins,
    color="#4C72B0",
    edgecolor="black",
    alpha=0.8,
    linewidth=0.7,
)

ax.set_title(chart_title, fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel(x_label, fontsize=14)
ax.set_ylabel(y_label, fontsize=14)
ax.grid(True, alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, None if y_auto else y_max)

# Bar labels
for patch in patches:
    height = patch.get_height()
    if height > 0:
        ax.annotate(
            f"{int(height)}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

plt.tight_layout()
st.pyplot(fig)

# ===========================
# Data table (optional)
# ===========================
with st.expander("View Raw Contour Values"):
    df = pd.DataFrame(value_file_pairs, columns=["Value", "Source File"])
    df = df.sort_values("Value").reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

# ===========================
# Download PNG
# ===========================
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
buf.seek(0)

st.download_button(
    label="Download Plot as PNG",
    data=buf,
    file_name="histogram.png",
    mime="image/png",
)

plt.close(fig)
