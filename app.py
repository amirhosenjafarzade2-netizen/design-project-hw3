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
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# 4. HEATMAP – ULTIMATE VERSION (Contour-Aware + Statistical + Detail Boost)
# ==================================================================
else:  # Heatmap – the one you've been waiting for
    with st.sidebar:
        st.subheader("Heatmap Mode")
        interp_mode = st.radio("Interpolation Style", 
                               ["Pure Contour-Aware (Exact & Fast)", 
                                "Statistical / Simulated (Creative)"])

        res = st.slider("Grid Resolution", 200, 1200, 800, 50)
        cmap = st.selectbox("Colormap", 
                            ["viridis", "plasma", "inferno", "turbo", "magma", "hot", "jet", "cividis", "RdBu_r", "coolwarm"], 
                            index=3)

        # ——————— Shared controls ———————
        detail_boost = st.slider("Detail Boost (Color Richness)", 0.3, 6.0, 1.5, 0.1,
                                 help="Higher = much more color variation inside bands → super detailed look")

        smoothness = st.slider("Final Smoothness", 0.0, 8.0, 2.2, 0.2)

        show_contours = st.checkbox("Overlay Original Contours", True)
        if show_contours:
            contour_lw = st.slider("Contour Line Width", 0.3, 2.5, 0.8, 0.1)

        st.subheader("Color Scale")
        auto_scale = st.checkbox("Auto Color Scale", True)
        if not auto_scale:
            col1, col2 = st.columns(2)
            with col1: vmin = st.number_input("Min Value", value=0.0)
            with col2: vmax = st.number_input("Max Value", value=1000.0)

        # ——————— Statistical mode only ———————
        if interp_mode == "Statistical / Simulated (Creative)":
            dist_type = st.selectbox("Distribution", ["Normal", "Lognormal", "Uniform"])
            variogram = st.selectbox("Variogram Range", ["Short", "Smooth", "Long"])

    from scipy.ndimage import gaussian_filter, distance_transform_edt
    from skimage.draw import polygon

    # =============== 1. Collect all data ===============
    contours_by_val = {}
    all_points = []
    raw_pts, raw_vals = [], []

    for f in uploaded_files:
        for val, pts in parse_bna(f):
            if len(pts) < 3: continue
            poly = np.array(pts)
            all_points.append(poly)
            contours_by_val.setdefault(val, []).append(poly)
            for x, y in pts:
                raw_pts.append([x, y])
                raw_vals.append(val)

    if not raw_pts:
        st.error("No contour data found.")
        st.stop()

    points = np.array(raw_pts)
    values = np.array(raw_vals)

    min_x = points[:,0].min(); max_x = points[:,0].max()
    min_y = points[:,1].min(); max_y = points[:,1].max()
    margin = 0.07 * max(max_x-min_x, max_y-min_y)
    min_x -= margin; max_x += margin; min_y -= margin; max_y += margin

    xi = np.linspace(min_x, max_x, res)
    yi = np.linspace(min_y, max_y, res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # =============== 2. Interpolation ===============
    if interp_mode == "Pure Contour-Aware (Exact & Fast)":
        # ——— Fast & perfect contour-aware ———
        mask = np.zeros((res, res), dtype=bool)
        for polys in contours_by_val.values():
            for poly in polys:
                px = np.interp(poly[:,0], [min_x, max_x], [0, res-1]).astype(int)
                py = np.interp(poly[:,1], [min_y, max_y], [res-1, 0]).astype(int)
                rr, cc = polygon(py, px, (res, res))
                for dr in range(-7, 8):
                    for dc in range(-7, 8):
                        if dr*dr + dc*dc <= 49:
                            r = np.clip(rr + dr, 0, res-1)
                            c = np.clip(cc + dc, 0, res-1)
                            mask[r, c] = True

        dist = distance_transform_edt(~mask)

        z_val = np.zeros((res, res))
        z_weight = np.zeros((res, res))

        for val, polys in contours_by_val.items():
            level_mask = np.zeros((res, res), dtype=bool)
            for poly in polys:
                px = np.interp(poly[:,0], [min_x, max_x], [0, res-1]).astype(int)
                py = np.interp(poly[:,1], [min_y, max_y], [res-1, 0]).astype(int)
                rr, cc = polygon(py, px, (res, res))
                for dr in range(-9, 10):
                    for dc in range(-9, 10):
                        if dr*dr + dc*dc <= 81:
                            r = np.clip(rr + dr, 0, res-1)
                            c = np.clip(cc + dc, 0, res-1)
                            level_mask[r, c] = True
            w = level_mask / (dist**2 + 1e-8)
            z_val += val * w
            z_weight += w

        grid_z = z_val / (z_weight + 1e-12)

    else:
        # ——— Your original statistical mode (but better) ———
        domain = max(max_x-min_x, max_y-min_y)
        corr_len_map = {"Short": 0.05, "Smooth": 0.22, "Long": 0.58}
        corr_len = corr_len_map[variogram] * domain

        # Simulate values if wanted
        if dist_type == "Normal":
            sim_vals = np.random.normal(values.mean(), values.std() or 1, len(values))
        elif dist_type == "Lognormal":
            pos = values[values > 0]; pos = pos if len(pos)>0 else values
            mu, sigma = np.log(pos).mean(), np.log(pos).std()
            sim_vals = np.exp(np.random.normal(mu, sigma, len(values)))
        else:
            sim_vals = np.random.uniform(values.min(), values.max(), len(values))

        # Exponential IDW (your original style)
        grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        dists = np.sqrt(((points[None,:,:] - grid_pts[:,None,:])**2).sum(-1))
        weights = np.exp(-dists / corr_len)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12
        grid_z = (weights * sim_vals[None,:]).sum(axis=1).reshape(res, res)

    # =============== 3. Post-processing ===============
    # Final smoothing
    if smoothness > 0:
        grid_z = gaussian_filter(grid_z, sigma=smoothness)

    # DETAIL BOOST — this is the magic you wanted!
    if detail_boost != 1.0:
        mean_val = np.nanmean(grid_z)
        grid_z = mean_val + (grid_z - mean_val) * detail_boost
        # Optional: power law for even more punch
        if detail_boost > 2.0:
            grid_z = np.sign(grid_z - mean_val) * np.abs(grid_z - mean_val)**0.8 + mean_val

    # =============== 4. Plot ===============
    fig, ax = plt.subplots(figsize=(14, 11), dpi=160)

    vmin_final = np.nanpercentile(grid_z, 2) if auto_scale else vmin
    vmax_final = np.nanpercentile(grid_z, 98) if auto_scale else vmax

    im = ax.imshow(grid_z, extent=[min_x, max_x, min_y, max_y], origin='lower',
                   cmap=cmap, vmin=vmin_final, vmax=vmax_final, interpolation='bilinear')

    if show_contours:
        for polys in contours_by_val.values():
            for poly in polys:
                ax.plot(poly[:,0], poly[:,1], color='white', lw=contour_lw, alpha=0.8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Value", rotation=270, labelpad=20)

    mode_suffix = "Contour-Aware" if interp_mode.startswith("Pure") else "Statistical"
    ax.set_title(f"{chart_title} – {mode_suffix} Heatmap", fontsize=16, pad=20)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect('equal')
    ax.grid(False)
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
