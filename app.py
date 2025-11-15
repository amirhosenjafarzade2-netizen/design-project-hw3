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
st.set_page_config(page_title="BNA Multi-Mode Analyzer", layout="wide")
st.title("BNA Contour Analyzer – 4 Modes")

# ------------------------------------------------------------------
# Helper: parse a BNA file → list of (value, [(x,y), …])
# ------------------------------------------------------------------
def parse_bna(file_obj):
    """Return list of (float_value, list_of_(x,y)_tuples)."""
    content = file_obj.read().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    contours = []          # [(value, [(x,y), …]), …]
    current_val = None
    current_pts = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # New contour header
        if line.startswith('"C"'):
            # Save previous
            if current_val is not None and current_pts:
                contours.append((current_val, current_pts.copy()))
            # Parse header
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    current_val = float(parts[1])   # <-- parameter label
                except ValueError:
                    current_val = None
                current_pts = []
            continue

        # Coordinate line
        coords = re.split(r'[,;\s]+', line)
        if len(coords) >= 2:
            try:
                x, y = float(coords[0]), float(coords[1])
                current_pts.append((x, y))
            except ValueError:
                continue

    # Save last contour
    if current_val is not None and current_pts:
        contours.append((current_val, current_pts))
    return contours

# ------------------------------------------------------------------
# File uploader (shared)
# ------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload BNA Files",
    type=["bna"],
    accept_multiple_files=True,
    help="Any number of .bna contour files"
)

if not uploaded_files:
    st.info("Upload at least one .bna file to start.")
    st.stop()

# ------------------------------------------------------------------
# Sidebar – global title / axis controls
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
    ["Histogram", "Contour Map (single file)", "Overlay Contours (two files)", "Heatmap (generated)"]
)

# ------------------------------------------------------------------
# 1. HISTOGRAM MODE
# ------------------------------------------------------------------
if mode == "Histogram":
    # ---- collect values (one per contour) ----
    all_vals = []
    pairs = []                     # (value, filename)
    for f in uploaded_files:
        contours = parse_bna(f)
        for val, _ in contours:
            all_vals.append(val)
            pairs.append((val, f.name))

    if not all_vals:
        st.error("No numeric contour values found.")
        st.stop()

    data = np.array(all_vals)

    # ---- controls ----
    with st.sidebar:
        st.subheader("Histogram")
        bins = st.slider("Number of Bins", 5, 200, 50, 5)
        x_min = st.number_input("X Min", value=float(data.min()), format="%.6g")
        x_max = st.number_input("X Max", value=float(data.max()), format="%.6g")
        y_auto = st.checkbox("Y Max Auto", True)
        y_max = st.number_input("Y Max", value=10.0, disabled=y_auto)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    counts, edges = np.histogram(data, bins=bins, range=(x_min, x_max))
    ax.bar(edges[:-1], counts, width=np.diff(edges),
           color="#4C72B0", edgecolor="black", alpha=0.85, linewidth=0.8)

    ax.set_title(chart_title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, None if y_auto else y_max)

    plt.tight_layout()
    st.pyplot(fig)

    # ---- table ----
    with st.expander("Raw Values"):
        df = pd.DataFrame(pairs, columns=["Value", "File"])
        st.dataframe(df.sort_values("Value"), use_container_width=True)

# ------------------------------------------------------------------
# 2. SINGLE CONTOUR MAP
# ------------------------------------------------------------------
elif mode == "Contour Map (single file)":
    file_options = {f.name: f for f in uploaded_files}
    chosen_name = st.selectbox("Choose BNA file", list(file_options.keys()))
    chosen_file = file_options[chosen_name]

    contours = parse_bna(chosen_file)

    # ---- controls ----
    with st.sidebar:
        st.subheader("Contour Settings")
        line_width = st.slider("Line Width", 0.5, 5.0, 1.5, 0.1)
        label_size = st.slider("Label Font Size", 6, 20, 10, 1)
        label_every = st.slider("Label every Nth line", 1, 20, 1, 1)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    for i, (val, pts) in enumerate(contours):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="black", linewidth=line_width)

        # label (only every Nth line)
        if label_every == 1 or i % label_every == 0:
            mid = len(pts) // 2
            ax.text(xs[mid], ys[mid], f"{val:.6g}",
                    fontsize=label_size, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_title(chart_title, fontsize=16, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------------
# 3. OVERLAY TWO CONTOURS
# ------------------------------------------------------------------
elif mode == "Overlay Contours (two files)":
    if len(uploaded_files) < 2:
        st.error("Need **at least two** BNA files for overlay.")
        st.stop()

    file_names = [f.name for f in uploaded_files]
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.selectbox("File A", file_names, index=0)
    with col2:
        file_b = st.selectbox("File B", file_names, index=1)

    fa = next(f for f in uploaded_files if f.name == file_a)
    fb = next(f for f in uploaded_files if f.name == file_b)

    contours_a = parse_bna(fa)
    contours_b = parse_bna(fb)

    # ---- controls ----
    with st.sidebar:
        st.subheader("Layer A")
        colA1, colA2 = st.columns(2)
        with colA1:
            color_a = st.color_picker("Color A", "#1f77b4")
        with colA2:
            alpha_a = st.slider("Opacity A", 0.0, 1.0, 0.7, 0.05, key="alpha_a")
        lw_a = st.slider("Line Width A", 0.5, 5.0, 1.2, 0.1, key="lw_a")
        label_a = st.checkbox("Show labels A", True)
        if label_a:
            label_every_a = st.slider("Label every Nth A", 1, 20, 3, key="lea")

        st.subheader("Layer B")
        colB1, colB2 = st.columns(2)
        with colB1:
            color_b = st.color_picker("Color B", "#ff7f0e")
        with colB2:
            alpha_b = st.slider("Opacity B", 0.0, 1.0, 0.7, 0.05, key="alpha_b")
        lw_b = st.slider("Line Width B", 0.5, 5.0, 1.2, 0.1, key="lw_b")
        label_b = st.checkbox("Show labels B", True)
        if label_b:
            label_every_b = st.slider("Label every Nth B", 1, 20, 3, key="leb")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Layer A
    for i, (val, pts) in enumerate(contours_a):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color_a, linewidth=lw_a, alpha=alpha_a)
        if label_a and (i % label_every_a == 0):
            mid = len(pts) // 2
            ax.text(xs[mid], ys[mid], f"{val:.6g}",
                    fontsize=9, color=color_a, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    # Layer B
    for i, (val, pts) in enumerate(contours_b):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color_b, linewidth=lw_b, alpha=alpha_b)
        if label_b and (i % label_every_b == 0):
            mid = len(pts) // 2
            ax.text(xs[mid], ys[mid], f"{val:.6g}",
                    fontsize=9, color=color_b, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    ax.set_title(chart_title, fontsize=16, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------------
# 4. HEATMAP (generated property map)
# ------------------------------------------------------------------
else:   # Heatmap
    with st.sidebar:
        st.subheader("Distribution")
        dist = st.selectbox("Distribution", ["Normal", "Lognormal", "Uniform"])

        st.subheader("Variogram / Correlation")
        vario = st.selectbox("Variogram behaviour", ["Smooth", "Short", "Long"])

        st.subheader("Grid")
        grid_res = st.slider("Grid resolution (points per side)", 50, 500, 200, 25)

        st.subheader("Property range")
        col1, col2 = st.columns(2)
        with col1:
            vmin = st.number_input("Min value", value=0.0)
        with col2:
            vmax = st.number_input("Max value", value=1.0)

        cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"])

    # ---- generate random points (simulate wells) ----
    np.random.seed(42)
    n_wells = 150
    x_w = np.random.uniform(0, 15000, n_wells)
    y_w = np.random.uniform(0, 15000, n_wells)

    # ---- assign property values according to distribution ----
    if dist == "Normal":
        values = np.clip(norm.rvs(loc=(vmin+vmax)/2, scale=(vmax-vmin)/6, size=n_wells), vmin, vmax)
    elif dist == "Lognormal":
        sigma = 0.5
        mu = np.log((vmin+vmax)/2) - 0.5*sigma**2
        values = np.clip(lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_wells), vmin, vmax)
    else:   # Uniform
        values = uniform.rvs(loc=vmin, scale=vmax-vmin, size=n_wells)

    # ---- variogram influence (correlation length) ----
    if vario == "Smooth":
        corr_len = 6000
    elif vario == "Short":
        corr_len = 1500
    else:   # Long
        corr_len = 12000

    # ---- simple exponential variogram interpolation (kriging-like) ----
    grid_x, grid_y = np.mgrid[0:15000:grid_res*1j, 0:15000:grid_res*1j]
    grid_flat = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # distance matrix
    dists = np.sqrt(((x_w[:, None] - grid_flat[:, 0])**2 +
                     (y_w[:, None] - grid_flat[:, 1])**2))

    # exponential covariance
    cov = np.exp(-dists / corr_len)
    weights = cov / cov.sum(axis=0)

    grid_z = np.dot(weights.T, values)
    grid_z = grid_z.reshape(grid_x.shape)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(grid_z, extent=[0, 15000, 0, 15000],
                   origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(chart_title, fontsize=16, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal', adjustable='box')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Property Value")

    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------------
# PNG download (works for every mode)
# ------------------------------------------------------------------
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
buf.seek(0)

st.download_button(
    label="Download Plot as PNG",
    data=buf,
    file_name="bna_analysis.png",
    mime="image/png"
)

# Clean up
if 'fig' in locals():
    plt.close(fig)
