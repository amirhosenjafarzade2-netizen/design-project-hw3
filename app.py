import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import io

# ===========================
# Page Configuration
# ===========================
st.set_page_config(page_title="BNA Contour Histogram", layout="wide")
st.title("BNA Contour Histogram Analyzer")
st.markdown("""
**Upload `.bna` files** → **See histograms of contour values (porosity, perm, NTG, etc.)**  
The **X-axis now shows the real parameter values** (e.g., `0.15`, `57.0`, `200.0`), not the internal count.
""")

# ===========================
# File Uploader
# ===========================
uploaded_files = st.file_uploader(
    "Upload BNA Contour Files",
    type=["bna"],
    accept_multiple_files=True,
    help="Upload one or more .bna files (Porosity_Contours.bna, Permeability_Contours.bna, etc.)"
)

if not uploaded_files:
    st.info("Please upload at least one `.bna` file to continue.")
    st.stop()

# ===========================
# BNA Parser – Uses SECOND FIELD (Parameter Label)
# ===========================
def parse_bna(file_obj):
    """
    Parses .bna file.
    Returns dict: parameter_value (float) → list of (x, y) points
    Uses the SECOND field after "C", e.g., "0.150" in "C","0.150",1191
    """
    content = file_obj.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    contours = {}
    current_value = None
    current_points = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # New contour: "C","label",count
        if line.startswith('"C"'):
            # Save previous contour
            if current_value is not None and current_points:
                contours[current_value] = current_points.copy()

            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 3:
                label = parts[1]  # <-- THIS IS THE REAL PARAMETER VALUE (e.g., "0.150")
                try:
                    value = float(label)
                except ValueError:
                    # If not a number, skip or keep as string (will be filtered later)
                    value = None
                current_value = value
                current_points = []
            continue

        # Coordinate line: X,Y
        coords = re.split(r'[,;\s]+', line)
        if len(coords) >= 2:
            try:
                x = float(coords[0])
                y = float(coords[1])
                current_points.append((x, y))
            except ValueError:
                continue

    # Save last contour
    if current_value is not None and current_points:
        contours[current_value] = current_points

    return contours

# ===========================
# Parse All Files & Collect Values + Source
# ===========================
value_file_pairs = []  # (value, filename)
all_values = []

for up_file in uploaded_files:
    try:
        contours = parse_bna(up_file)
        for val in contours.keys():
            if isinstance(val, (int, float)) and val is not None:
                numeric_val = float(val)
                value_file_pairs.append((numeric_val, up_file.name))
                all_values.append(numeric_val)
    except Exception as e:
        st.error(f"Error parsing **{up_file.name}**: {e}")

if not all_values:
    st.error("No valid numeric contour values found. Check file format.")
    st.stop()

data = np.array(all_values)

# ===========================
# Sidebar Controls
# ===========================
with st.sidebar:
    st.header("Histogram Settings")

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Values", len(data))
        st.metric("Min", f"{data.min():.6g}")
    with col2:
        st.metric("Max", f"{data.max():.6g}")
        st.metric("Mean", f"{data.mean():.6g}")

    st.markdown("---")

    num_bins = st.slider("Number of Bins", min_value=5, max_value=200, value=50, step=5)
    chart_title = st.text_input("Chart Title", value="Histogram of Contour Values")
    x_label = st.text_input("X-Axis Label", value="Parameter Value")
    y_label = st.text_input("Y-Axis Label", value="Frequency")

    # Axis limits
    col1, col2 = st.columns(2)
    with col1:
        x_min = st.number_input("X Min", value=float(data.min()), format="%.6g")
        y_min = st.number_input("Y Min", value=0.0)
    with col2:
        x_max = st.number_input("X Max", value=float(data.max()), format="%.6g")
        y_auto = st.checkbox("Y Max Auto", value=True)
        y_max = st.number_input("Y Max", value=float(int(data.max()) + 1), disabled=y_auto)

# ===========================
# Create Histogram
# ===========================
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# Histogram
n, bins, patches = ax.hist(
    data,
    bins=num_bins,
    color="#4C72B0",
    edgecolor="black",
    alpha=0.85,
    linewidth=0.8,
    rwidth=0.95
)

# Styling
ax.set_title(chart_title, fontsize=16, fontweight="bold", pad=20)
ax.set_xlabel(x_label, fontsize=14)
ax.set_ylabel(y_label, fontsize=14)
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
ax.set_axisbelow(True)

# Axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, None if y_auto else y_max)

# Add count labels on top of bars
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
            weight="bold"
        )

plt.tight_layout()

# ===========================
# Display Plot
# ===========================
st.pyplot(fig)

# ===========================
# Raw Data Table
# ===========================
with st.expander("View Raw Contour Values"):
    df = pd.DataFrame(value_file_pairs, columns=["Value", "Source File"])
    df = df.sort_values("Value").reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

# ===========================
# Download Button
# ===========================
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
buf.seek(0)

st.download_button(
    label="Download Plot as PNG",
    data=buf,
    file_name="contour_histogram.png",
    mime="image/png"
)

# Free memory
plt.close(fig)

# ===========================
# Footer
# ===========================
st.caption("Built with Streamlit • Matplotlib • Pandas | Fixed: Uses **parameter label** (2nd field) for X-axis")
