import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import re

# ===========================
# App Title
# ===========================
st.set_page_config(page_title="BNA Contour Histogram Analyzer", layout="wide")
st.title("🗺️ BNA Contour Histogram Analyzer")
st.markdown("""
Upload `.bna` contour files (e.g., Porosity, Permeability, NTG, Thickness, etc.)  
and generate customizable high-resolution histograms.
""")

# ===========================
# File Uploader
# ===========================
uploaded_files = st.file_uploader(
    "Upload BNA Files",
    type=["bna"],
    accept_multiple_files=True,
    help="Upload one or more .bna contour files"
)

if not uploaded_files:
    st.info("Please upload at least one .bna file to begin.")
    st.stop()

# ===========================
# Parse BNA Files
# ===========================
def parse_bna(file_obj):
    content = file_obj.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()
    
    contours = {}
    current_label = None
    current_points = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('"C"'):
            # Save previous contour
            if current_label is not None and current_points:
                contours[current_label] = current_points.copy()
            # Parse new contour
            parts = line.split(',')
            if len(parts) >= 3:
                label = parts[1].strip('"')
                try:
                    value = float(parts[2])
                except:
                    value = label  # fallback
                current_label = value
                current_points = []
        else:
            # Parse X,Y coordinate
            coords = re.split(r'[,;\s]+', line.strip())
            if len(coords) >= 2:
                try:
                    x = float(coords[0])
                    y = float(coords[1])
                    current_points.append((x, y))
                except:
                    continue
    # Save last contour
    if current_label is not None and current_points:
        contours[current_label] = current_points
    
    return contours

# Extract all contour values (keys) from all files
all_contour_values = []
file_names = []

for file in uploaded_files:
    try:
        contours = parse_bna(file)
        values = [float(k) for k in contours.keys() if isinstance(k, (int, float)) or k.replace('.','').isdigit()]
        all_contour_values.extend(values)
        file_names.append(file.name)
    except Exception as e:
        st.error(f"Error parsing {file.name}: {e}")

if not all_contour_values:
    st.error("No valid contour values found in uploaded files.")
    st.stop()

# ===========================
# Sidebar Controls
# ===========================
with st.sidebar:
    st.header("📊 Histogram Settings")
    
    # Combine all data
    data = np.array(all_contour_values)
    
    # Stats
    st.metric("Total Contour Values", len(data))
    st.metric("Min", f"{data.min():.4f}")
    st.metric("Max", f"{data.max():.4f}")
    st.metric("Mean", f"{data.mean():.4f}")
    
    st.markdown("---")
    
    # User inputs
    num_bins = st.slider("Number of Bins", min_value=5, max_value=200, value=50, step=5)
    
    chart_title = st.text_input("Chart Title", value="Histogram of Contour Values")
    
    x_label = st.text_input("X-Axis Label", value="Contour Value")
    y_label = st.text_input("Y-Axis Label", value="Frequency")
    
    # Manual axis limits
    col1, col2 = st.columns(2)
    with col1:
        x_min = st.number_input("X Min", value=float(data.min()), format="%.6f")
        y_min = st.number_input("Y Min", value=0.0)
    with col2:
        x_max = st.number_input("X Max", value=float(data.max()), format="%.6f")
        y_max_auto = st.checkbox("Y Max Auto", value=True)
        y_max = st.number_input("Y Max", value=float(data.max()), disabled=y_max_auto)

# ===========================
# Plot Histogram
# ===========================
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

# Histogram
n, bins, patches = ax.hist(
    data,
    bins=num_bins,
    color='#4C72B0',
    edgecolor='black',
    alpha=0.8,
    linewidth=0.8
)

# Styling
ax.set_title(chart_title, fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel(x_label, fontsize=14)
ax.set_ylabel(y_label, fontsize=14)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Set axis limits
ax.set_xlim(x_min, x_max)
if y_max_auto:
    ax.set_ylim(0, None)
else:
    ax.set_ylim(y_min, y_max)

# Add value labels on top of bars
for rect in patches:
    height = rect.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9, color='black')

# Tight layout
plt.tight_layout()

# ===========================
# Display Plot
# ===========================
st.pyplot(fig)

# ===========================
# Data Table (Optional)
# ===========================
with st.expander("📋 View Raw Contour Values"):
    df = pd.DataFrame({
        "Value": all_contour_values,
        "Source File": [f for f in file_names for _ in range(len([v for v in parse_bna(uploaded_files[file_names.index(f)]).keys() if isinstance(v, (int,float)) or str(v).replace('.','').isdigit()]))]
    })
    st.dataframe(df.sort_values("Value"), use_container_width=True)

# ===========================
# Download Plot
# ===========================
buf = plt.gcf().canvas.tostring_rgb()
import io
img_bytes = io.BytesIO()
fig.savefig(img_bytes, format='png', dpi=200, bbox_inches='tight')
img_bytes.seek(0)

st.download_button(
    label="📥 Download Plot as PNG",
    data=img_bytes,
    file_name="histogram.png",
    mime="image/png"
)

# Close plot to free memory
plt.close(fig)
