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
            if len(pts) >= 3:
                prev = mid - 1
                next_ = mid + 1
                dx = xs[next_] - xs[prev]
                dy = ys[next_] - ys[prev]
                angle = np.arctan2(dy, dx) * 180 / np.pi
            else:
                angle = 0
            ax.text(xs[mid], ys[mid], f"{val:.6g}", fontsize=label_size,
                    ha="center", va="center", rotation=angle,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

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
        ca_dashed = st.checkbox("Dashed Lines A", False, key="a_dashed")
        ca_label = st.checkbox("Labels A", True)
        if ca_label:
            ca_every = st.slider("Every Nth A", 1, 20, 3, key="a_every")
            ca_fontsize = st.slider("Font Size A", 6, 20, 10, key="a_font")

        st.subheader("Layer B")
        cb_color = st.color_picker("Color B", "#ff7f0e")
        cb_alpha = st.slider("Opacity B", 0.0, 1.0, 0.7, key="b_alpha")
        cb_lw = st.slider("Width B", 0.5, 5.0, 1.2, key="b_lw")
        cb_dashed = st.checkbox("Dashed Lines B", False, key="b_dashed")
        cb_label = st.checkbox("Labels B", True)
        if cb_label:
            cb_every = st.slider("Every Nth B", 1, 20, 3, key="b_every")
            cb_fontsize = st.slider("Font Size B", 6, 20, 10, key="b_font")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    for i, (val, pts) in enumerate(ca):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=ca_color, lw=ca_lw, alpha=ca_alpha, linestyle='--' if ca_dashed else '-')
        if ca_label and (i % ca_every == 0):
            mid = len(pts)//2
            if len(pts) >= 3:
                prev = mid - 1
                next_ = mid + 1
                dx = xs[next_] - xs[prev]
                dy = ys[next_] - ys[prev]
                angle = np.arctan2(dy, dx) * 180 / np.pi
            else:
                angle = 0
            ax.text(xs[mid], ys[mid], f"{val:.6g}", color=ca_color, fontsize=ca_fontsize,
                    ha="center", va="center", rotation=angle, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    for i, (val, pts) in enumerate(cb):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=cb_color, lw=cb_lw, alpha=cb_alpha, linestyle='--' if cb_dashed else '-')
        if cb_label and (i % cb_every == 0):
            mid = len(pts)//2
            if len(pts) >= 3:
                prev = mid - 1
                next_ = mid + 1
                dx = xs[next_] - xs[prev]
                dy = ys[next_] - ys[prev]
                angle = np.arctan2(dy, dx) * 180 / np.pi
            else:
                angle = 0
            ax.text(xs[mid], ys[mid], f"{val:.6g}", color=cb_color, fontsize=cb_fontsize,
                    ha="center", va="center", rotation=angle, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

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
    # ---------- Sidebar controls ----------
    with st.sidebar:
        st.subheader("Data source")
        use_real = st.checkbox("Use real contour points (recommended)", value=True)

        # ---- ALWAYS show distribution & variogram ----
        st.subheader("Statistical Model")
        dist = st.selectbox(
            "Distribution",
            ["Normal", "Lognormal", "Uniform"],
            key="dist"
        )
        vario = st.selectbox(
            "Variogram Behavior",
            ["Smooth", "Short", "Long"],
            key="vario"
        )

        # ---- Grid & colormap ----
        st.subheader("Grid")
        res = st.slider("Resolution (cells)", 50, 500, 200, 25, key="res")
        cmap = st.selectbox(
            "Colormap",
            ["viridis", "plasma", "inferno", "hot", "jet", "turbo"],
            key="cmap"
        )

        # ---- Real-data specific options ----
        if use_real:
            interp_method = st.selectbox(
                "Fallback Interpolation (griddata)",
                ["linear", "cubic", "nearest"],
                index=0,
                key="interp"
            )
            use_statistical = st.checkbox(
                "Use statistical smoothing (instead of griddata)",
                value=False
            )
        else:
            use_statistical = True   # forced when simulating

    # ---------- 1. Gather points from BNA files ----------
    raw_points, raw_values = [], []
    for f in uploaded_files:
        for val, pts in parse_bna(f):
            for x, y in pts:
                raw_points.append([x, y])
                raw_values.append(val)
    if not raw_points:
        st.error("No contour points found in the uploaded BNA files.")
        st.stop()

    points = np.array(raw_points)
    values = np.array(raw_values)

    # ---------- 2. Grid extent ----------
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    margin = 0.05 * max(max_x - min_x, max_y - min_y)
    min_x, max_x = min_x - margin, max_x + margin
    min_y, max_y = min_y - margin, max_y + margin

    grid_x, grid_y = np.mgrid[min_x:max_x:res*1j, min_y:max_y:res*1j]
    grid_shape = grid_x.shape
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # ---------- 3. Correlation length ----------
    domain = max(max_x - min_x, max_y - min_y)
    corr_len = {"Smooth": 0.30, "Short": 0.08, "Long": 0.60}[vario] * domain

    # ---------- 4. Interpolation / Simulation ----------
    if use_real and not use_statistical:
        # ---- Classic griddata (fast, exact) ----
        grid_z = griddata(points, values, (grid_x, grid_y), method=interp_method)
        grid_z = np.nan_to_num(grid_z, nan=np.nanmean(grid_z))
        title_suffix = " (real – griddata)"

    else:
        # ---- Statistical smoothing (real or simulated) ----
        if use_real:
            seed_pts, seed_val = points, values
            sim_mode = "real-smoothed"
        else:
            n_seeds = min(200, len(points))
            idx = np.random.choice(len(points), n_seeds, replace=False)
            seed_pts, seed_val = points[idx], values[idx]
            sim_mode = "simulated"

        # ---- Values (real or generated) ----
        if not use_real:
            if dist == "Normal":
                sim_vals = np.random.normal(
                    loc=seed_val.mean(),
                    scale=seed_val.std() or 1.0,
                    size=len(seed_val)
                )
            elif dist == "Lognormal":
                pos = seed_val[seed_val > 0]
                pos = pos if len(pos) else np.array([1.0])
                mu, s = np.log(pos).mean(), np.log(pos).std()
                sim_vals = np.exp(np.random.normal(mu, s, size=len(seed_val)))
            else:  # Uniform
                sim_vals = np.random.uniform(
                    seed_val.min(), seed_val.max(), size=len(seed_val)
                )
        else:
            sim_vals = seed_val   # keep real values

        # ---- Exponential covariance weighting ----
        dists = np.sqrt(((seed_pts[None, :, :] - grid_pts[:, None, :]) ** 2).sum(-1))
        weights = np.exp(-dists / corr_len)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12
        grid_z = (weights * sim_vals[None, :]).sum(axis=1)
        grid_z = grid_z.reshape(grid_shape)

        title_suffix = (
            f" ({'real-smoothed' if use_real else 'simulated'} – {dist}/{vario})"
        )

    # ---------- 5. Plot ----------
    vmin, vmax = np.nanmin(grid_z), np.nanmax(grid_z)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(
        grid_z,
        extent=[min_x, max_x, min_y, max_y],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(chart_title + title_suffix)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Value")
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
