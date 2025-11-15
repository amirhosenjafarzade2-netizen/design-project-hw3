import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import io
from scipy.interpolate import splprep, splev
from scipy.stats import linregress, norm, lognorm, uniform

# ===========================
# Page config
# ===========================
st.set_page_config(page_title="BNA Analyzer – Final", layout="wide")
st.title("BNA Contour Analyzer – Final & Perfect")
st.markdown("**Histogram | Contour Map | Overlay (Fault Fixed) | Heatmap (Auto Scale)**")

# ------------------------------------------------------------------
# BNA Parser
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
    help="Porosity_Contours.bna, Structural_Contours.bna, etc."
)

if not uploaded_files:
    st.info("Upload at least one .bna file to begin.")
    st.stop()

# ------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Global Settings")
    chart_title = st.text_input("Chart Title", "Reservoir Analysis")
    x_label = st.text_input("X-Axis Label", "X (m)")
    y_label = st.text_input("Y-Axis Label", "Y (m)")

# ------------------------------------------------------------------
# Mode selector
# ------------------------------------------------------------------
mode = st.radio("Select Mode", ["Histogram", "Contour Map", "Overlay Contours", "Heatmap"])

# ==================================================================
# 1. HISTOGRAM – CORRECT BINNING
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
        st.error("No numeric contour values found.")
        st.stop()

    data = np.array(all_vals)

    with st.sidebar:
        st.subheader("Histogram Settings")
        num_bins = st.slider("Number of Bins", 5, 200, 50, 5)
        x_min = st.number_input("X Min", value=float(data.min()), format="%.6g")
        x_max = st.number_input("X Max", value=float(data.max()), format="%.6g")
        y_auto = st.checkbox("Y Max Auto", True)
        y_max = st.number_input("Y Max", value=10.0, disabled=y_auto)

    # Filter data in range
    mask = (data >= x_min) & (data <= x_max)
    data_in = data[mask]
    if len(data_in) == 0:
        st.warning("No data in selected X range.")
        st.stop()

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    ax.hist(data_in, bins=num_bins, range=(x_min, x_max),
            color="#4C72B0", edgecolor="black", alpha=0.85, linewidth=0.8)
    ax.set_title(chart_title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, None if y_auto else y_max)
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("Raw Contour Values"):
        df = pd.DataFrame(pairs, columns=["Value", "Source File"])
        df = df.sort_values("Value").reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total: {len(df)} contours")

# ==================================================================
# 2. CONTOUR MAP (Single File)
# ==================================================================
elif mode == "Contour Map":
    file_dict = {f.name: f for f in uploaded_files}
    chosen = st.selectbox("Select BNA File", list(file_dict.keys()))
    contours = parse_bna(file_dict[chosen])

    with st.sidebar:
        st.subheader("Contour Settings")
        line_width = st.slider("Line Width", 0.5, 5.0, 1.5, 0.1)
        label_size = st.slider("Label Font Size", 6, 20, 10)
        label_every = st.slider("Label Every Nth Line", 1, 20, 1)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    for i, (val, pts) in enumerate(contours):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "k-", linewidth=line_width)
        if label_every == 1 or i % label_every == 0:
            mid = len(pts) // 2
            ax.text(xs[mid], ys[mid], f"{val:.6g}", fontsize=label_size,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_title(chart_title)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig)

# ==================================================================
# 3. OVERLAY CONTOURS – FAULT LINE FIXED & SMOOTH
# ==================================================================
elif mode == "Overlay Contours":
    if len(uploaded_files) < 2:
        st.error("Upload at least 2 BNA files.")
        st.stop()

    names = [f.name for f in uploaded_files]
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.selectbox("Layer A (Contours)", names, 0)
    with col2:
        file_b = st.selectbox("Layer B (Contours)", names, 1)

    fa = next(f for f in uploaded_files if f.name == file_a)
    fb = next(f for f in uploaded_files if f.name == file_b)
    ca = parse_bna(fa)
    cb = parse_bna(fb)
    all_contours = ca + cb

    with st.sidebar:
        st.subheader("Layer A")
        color_a = st.color_picker("Color A", "#1f77b4")
        alpha_a = st.slider("Opacity A", 0.0, 1.0, 0.7)
        lw_a = st.slider("Width A", 0.5, 5.0, 1.2)
        label_a = st.checkbox("Show Labels A", True)
        if label_a:
            every_a = st.slider("Label Every Nth A", 1, 20, 3)

        st.subheader("Layer B")
        color_b = st.color_picker("Color B", "#ff7f0e")
        alpha_b = st.slider("Opacity B", 0.0, 1.0, 0.7)
        lw_b = st.slider("Width B", 0.5, 5.0, 1.2)
        label_b = st.checkbox("Show Labels B", True)
        if label_b:
            every_b = st.slider("Label Every Nth B", 1, 20, 3)

        st.subheader("Fault / Slope Line")
        fault_color = st.color_picker("Fault Line Color", "#d62728")
        fault_width = st.slider("Fault Line Width", 1.0, 6.0, 2.5, 0.5)
        fault_style = st.selectbox("Fault Line Style", ["Solid", "Dashed", "Dotted", "Dash-dot"])

    style_map = {"Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."}
    dash = style_map[fault_style]

    # ------------------- FAULT DETECTION -------------------
    def is_closed(pts):
        if len(pts) < 3: return False
        return abs(pts[0][0] - pts[-1][0]) < 1 and abs(pts[0][1] - pts[-1][1]) < 1

    def linearity(pts):
        xs, ys = zip(*pts)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 3: return 0
        _, _, r, _, _ = linregress(xs, ys)
        return r**2

    def slope_angle(pts):
        xs, ys = zip(*pts)
        dx = abs(max(xs) - min(xs))
        dy = abs(max(ys) - min(ys))
        return np.degrees(np.arctan2(dy, dx)) if dx > 0 else 90

    def line_length(pts):
        xs, ys = zip(*pts)
        return np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))

    candidates = []
    for val, pts in all_contours:
        if is_closed(pts): continue
        if len(pts) < 10: continue
        r2 = linearity(pts)
        angle = slope_angle(pts)
        length = line_length(pts)
        if r2 > 0.95 and 20 < angle < 70 and length > 3000:
            score = r2 * (length / 1000) * (1 / (abs(angle - 45) + 1))
            candidates.append((score, val, pts))

    fault_contour = None
    if candidates:
        candidates.sort(reverse=True)
        fault_contour = candidates[0][1:]  # (val, pts)

    # ------------------- SMOOTHING -------------------
    def smooth_line(pts, n=400):
        xs, ys = zip(*pts)
        xs, ys = np.array(xs), np.array(ys)
        mask = np.append([True], np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) > 1)
        xs, ys = xs[mask], ys[mask]
        if len(xs) < 4:
            return xs, ys
        try:
            tck, u = splprep([xs, ys], s=0, k=3)
            u_new = np.linspace(u.min(), u.max(), n)
            return splev(u_new, tck)
        except:
            return xs, ys

    # ------------------- PLOT -------------------
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Layer A
    for i, (val, pts) in enumerate(ca):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color_a, lw=lw_a, alpha=alpha_a)
        if label_a and (i % every_a == 0):
            mid = len(pts)//2
            ax.text(xs[mid], ys[mid], f"{val:.3f}", color=color_a, fontsize=9,
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Layer B
    for i, (val, pts) in enumerate(cb):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color_b, lw=lw_b, alpha=alpha_b)
        if label_b and (i % every_b == 0):
            mid = len(pts)//2
            ax.text(xs[mid], ys[mid], f"{val:.3f}", color=color_b, fontsize=9,
                    ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Fault Line
    if fault_contour:
        val, pts = fault_contour
        x_s, y_s = smooth_line(pts)
        ax.plot(x_s, y_s, color=fault_color, lw=fault_width, linestyle=dash,
                label=f"Fault Line ({val:.3f})", alpha=0.95)
        ax.legend(loc="upper right")

    ax.set_title(chart_title)
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig)

# ==================================================================
# 4. HEATMAP – AUTO COLOR SCALE
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

    if dist == "Normal":
        values = np.random.normal(0.5, 0.15, n_wells)
    elif dist == "Lognormal":
        values = lognorm.rvs(s=0.5, scale=0.4, size=n_wells)
    else:
        values = np.random.uniform(0, 1, n_wells)
    values = np.clip(values, 0, 1)

    corr_len = {"Smooth": 6000, "Short": 1500, "Long": 12000}[vario]
    grid_x, grid_y = np.mgrid[0:15000:res*1j, 0:15000:res*1j]
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dists = np.sqrt((x_w[None,:] - points[:,0,None])**2 + (y_w[None,:] - points[:,1,None])**2)
    weights = np.exp(-dists / corr_len)
    weights /= weights.sum(axis=1, keepdims=True)
    grid_z = np.dot(weights, values).reshape(grid_x.shape)

    vmin, vmax = grid_z.min(), grid_z.max()

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(grid_z, extent=[0,15000,0,15000], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(chart_title); ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Property Value")
    plt.tight_layout(); st.pyplot(fig)

# ==================================================================
# PNG Download
# ==================================================================
buf = io.BytesIO()
if 'fig' in locals():
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    st.download_button(
        label="Download Plot as PNG",
        data=buf,
        file_name="bna_analysis.png",
        mime="image/png"
    )
    plt.close(fig)

# ==================================================================
# Footer
# ==================================================================
st.caption("BNA Analyzer – Final Version | Fault Fixed | Auto Color | Streamlit + Matplotlib")
