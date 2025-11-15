import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import io
from matplotlib import colors
from scipy.interpolate import griddata
from scipy.stats import norm, lognorm, uniform

# ===========================
# Page config
# ===========================
st.set_page_config(page_title="BNA Analyzer – All Modes", layout="wide")
st.title("BNA Contour Analyzer – 4 Perfect Modes")

# ------------------------------------------------------------------
# Helper: parse BNA → list of (value, [(x,y)])
# ------------------------------------------------------------------
def parse_bna(file_obj):
    content = file_obj.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()
    contours = []
    current_val = None
    current_pts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('"C"'):
            if current_val is not None and current_pts:
                contours.append((current_val, current_pts.copy()))
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    current_val = float(parts[1])
                except ValueError:
                    current_val = None
                current_pts = []
            continue
        coords = re.split(r'[,;\s]+', line)
        if len(coords) >= 2:
            try:
                x, y = float(coords[0]), float(coords[1])
                current_pts.append((x, y))
            except ValueError:
                continue
    if current_val is not None and current_pts:
        contours.append((current_val, current_pts))
    return contours

# ------------------------------------------------------------------
# File uploader
# ------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload BNA Files",
    type=["bna"],
    accept_multiple_files=True,
    help="Any .bna contour files"
)

if not uploaded_files:
    st.info("Upload at least one .bna file.")
    st.stop()

# ------------------------------------------------------------------
# Global title / axis
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Global Settings")
    chart_title = st.text_input("Chart Title", "BNA Analysis")
    x_label = st.text_input("X-Axis Label", "X (m)")
    y_label = st.text_input("Y-Axis Label", "Y (m)")

# ------------------------------------------------------------------
# Mode selector
# ------------------------------------------------------------------
mode = st.radio(
    "Select Mode",
    ["Histogram", "Contour Map", "Overlay Contours", "Heatmap (Generated)"]
)

# ==================================================================
# 1. HISTOGRAM – FIXED BINNING (like your old correct version)
# ==================================================================
if mode == "Histogram":
    all_vals = []
    pairs = []
    for f in uploaded_files:
        contours = parse_bna(f)
        for val, _ in contours:
            all_vals.append(val)
            pairs.append((val, f.name))

    if not all_vals:
        st.error("No numeric values found.")
        st.stop()

    data = np.array(all_vals)

    # ---- Sidebar controls ----
    with st.sidebar:
        st.subheader("Histogram")
        num_bins = st.slider("Bins", 5, 200, 50, 5)
        x_min = st.number_input("X Min", value=float(data.min()), format="%.6g")
        x_max = st.number_input("X Max", value=float(data.max()), format="%.6g")
        y_auto = st.checkbox("Y Max Auto", True)
        y_max = st.number_input("Y Max", value=10.0, disabled=y_auto)

    # ---- Filter data in range BEFORE binning (critical!) ----
    mask = (data >= x_min) & (data <= x_max)
    data_in_range = data[mask]

    if len(data_in_range) == 0:
        st.warning("No data in selected X range.")
        st.stop()

    # ---- Correct histogram (same as your old working version) ----
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    counts, bin_edges, _ = ax.hist(
        data_in_range,
        bins=num_bins,
        range=(x_min, x_max),
        color="#4C72B0",
        edgecolor="black",
        alpha=0.85,
        linewidth=0.8
    )

    ax.set_title(chart_title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, None if y_auto else y_max)

    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("Raw Values"):
        df = pd.DataFrame(pairs, columns=["Value", "File"])
        st.dataframe(df.sort_values("Value"), use_container_width=True)

# ==================================================================
# 2. CONTOUR MAP (single file)
# ==================================================================
elif mode == "Contour Map":
    file_names = {f.name: f for f in uploaded_files}
    chosen = st.selectbox("Select file", list(file_names.keys()))
    contours = parse_bna(file_names[chosen])

    with st.sidebar:
        st.subheader("Contour")
        lw = st.slider("Line Width", 0.5, 5.0, 1.5, 0.1)
        label_size = st.slider("Label Size", 6, 20, 10)
        every_n = st.slider("Label every Nth", 1, 20, 1)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    for i, (val, pts) in enumerate(contours):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "k-", linewidth=lw)
        if every_n == 1 or i % every_n == 0:
            mid = len(pts) // 2
            ax.text(xs[mid], ys[mid], f"{val:.6g}", fontsize=label_size,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    ax.set_title(chart_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ==================================================================
# 3. OVERLAY CONTOURS
# ==================================================================
elif mode == "Overlay Contours":
    if len(uploaded_files) < 2:
        st.error("Need 2+ files.")
        st.stop()

    names = [f.name for f in uploaded_files]
    col1, col2 = st.columns(2)
    with col1:
        a = st.selectbox("Layer A", names, 0)
    with col2:
        b = st.selectbox("Layer B", names, 1)

    fa = next(f for f in uploaded_files if f.name == a)
    fb = next(f for f in uploaded_files if f.name == b)
    ca = parse_bna(fa)
    cb = parse_bna(fb)

    with st.sidebar:
        st.subheader("Layer A")
        ca_color = st.color_picker("Color A", "#1f77b4")
        ca_alpha = st.slider("Opacity A", 0.0, 1.0, 0.7, key="a_alpha")
        ca_lw = st.slider("Width A", 0.5, 5.0, 1.2, key="a_lw")
        ca_label = st.checkbox("Labels A", True)
        if ca_label:
            ca_every = st.slider("Every Nth A", 1, 20, 3, key="a_every")

        st.subheader("Layer B")
        cb_color = st.color_picker("Color B", "#ff7f0e")
        cb_alpha = st.slider("Opacity B", 0.0, 1.0, 0.7, key="b_alpha")
        cb_lw = st.slider("Width B", 0.5, 5.0, 1.2, key="b_lw")
        cb_label = st.checkbox("Labels B", True)
        if cb_label:
            cb_every = st.slider("Every Nth B", 1, 20, 3, key="b_every")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    for i, (val, pts) in enumerate(ca):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=ca_color, lw=ca_lw, alpha=ca_alpha)
        if ca_label and (i % ca_every == 0):
            mid = len(pts)//2
            ax.text(xs[mid], ys[mid], f"{val:.6g}", color=ca_color, fontsize=9,
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    for i, (val, pts) in enumerate(cb):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=cb_color, lw=cb_lw, alpha=cb_alpha)
        if cb_label and (i % cb_every == 0):
            mid = len(pts)//2
            ax.text(xs[mid], ys[mid], f"{val:.6g}", color=cb_color, fontsize=9,
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_title(chart_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ==================================================================
# 4. HEATMAP – AUTO color scale from data min/max
# ==================================================================
else:  # Heatmap
    with st.sidebar:
        st.subheader("Distribution")
        dist = st.selectbox("Type", ["Normal", "Lognormal", "Uniform"])
        st.subheader("Variogram")
        vario = st.selectbox("Behavior", ["Smooth", "Short", "Long"])
        st.subheader("Grid")
        res = st.slider("Resolution", 50, 500, 200, 25)
        cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "hot", "jet", "turbo"])

    np.random.seed(42)
    n_wells = 150
    x_w = np.random.uniform(0, 15000, n_wells)
    y_w = np.random.uniform(0, 15000, n_wells)

    # Generate values
    if dist == "Normal":
        values = np.random.normal(0.5, 0.15, n_wells)
    elif dist == "Lognormal":
        values = lognorm.rvs(s=0.5, scale=0.4, size=n_wells)
    else:
        values = np.random.uniform(0, 1, n_wells)

    # Clip to [0,1] for consistency
    values = np.clip(values, 0, 1)

    # Variogram length
    corr_len = {"Smooth": 6000, "Short": 1500, "Long": 12000}[vario]

    # Grid
    grid_x, grid_y = np.mgrid[0:15000:res*1j, 0:15000:res*1j]
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Distance + exponential weights
    dists = np.sqrt((x_w[None,:] - points[:,0,None])**2 + (y_w[None,:] - points[:,1,None])**2)
    weights = np.exp(-dists / corr_len)
    weights /= weights.sum(axis=1, keepdims=True)
    grid_z = np.dot(weights, values).reshape(grid_x.shape)

    # AUTO vmin/vmax from data
    vmin, vmax = grid_z.min(), grid_z.max()

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(grid_z, extent=[0,15000,0,15000], origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(chart_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Property Value")

    plt.tight_layout()
    st.pyplot(fig)

# ==================================================================
# PNG Download
# ==================================================================
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
buf.seek(0)
st.download_button("Download PNG", buf, "bna_plot.png", "image/png")
if 'fig' in locals():
    plt.close(fig)
