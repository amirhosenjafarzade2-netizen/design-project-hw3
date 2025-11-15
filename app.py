import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial.distance import pdist, squareform
import os
import io

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(page_title="BNA Geostat Analyzer", layout="wide")
st.title("BNA File Analyzer: Contour, Variogram, Histogram & Overlay")
st.markdown("Upload `.bna` → Choose **map type per file** → Customize **axes, title, levels** → Generate")

# ========================================
# MAP TYPES
# ========================================
MAP_TYPES = [
    "Contour Map",
    "Heatmap (Net Zero Contour)",
    "Histogram",
    "Normal Variogram",
    "Long/Short Directional Variogram",
    "Uniform Long Variogram",
    "Overlay: Structure + Thickness"
]

# ========================================
# PARSING .bna
# ========================================
def parse_bna(content):
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    contours = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('"C"'):
            parts = [p.strip('"') for p in line.split(',')]
            if len(parts) < 3: 
                i += 1
                continue
            try:
                z_value = float(parts[1])
                n_points = int(parts[2])
            except:
                i += 1
                continue
            points = []
            for j in range(1, min(n_points + 1, len(lines) - i)):
                coord_line = lines[i + j]
                if ',' in coord_line:
                    x, y = coord_line.split(',')
                    try:
                        points.append((float(x), float(y)))
                    except:
                        continue
            if len(points) >= 3:
                contours.append({'z': z_value, 'points': np.array(points)})
            i += n_points
        i += 1
    return contours

# ========================================
# DATAFRAME FROM CONTOURS
# ========================================
def contours_to_df(contours):
    all_x, all_y, all_z = [], [], []
    for c in contours:
        xs, ys = c['points'][:, 0], c['points'][:, 1]
        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend([c['z']] * len(xs))
    return pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})

# ========================================
# PLOTTING FUNCTIONS
# ========================================
def add_north_arrow(ax):
    ax.annotate('N', xy=(0.92, 0.92), xycoords='axes fraction',
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", edgecolor="black"))
    ax.arrow(0.92, 0.87, 0, 0.04, head_width=0.02, head_length=0.03,
             fc='black', ec='black', transform=ax.transAxes, linewidth=1.5)

def add_scale_bar(ax, length=2000):
    xlim = ax.get_xlim()
    scale = length / (xlim[1] - xlim[0])
    x0, x1 = 0.1, 0.1 + scale
    y0 = 0.05
    ax.plot([x0, x1], [y0, y0], color='black', linewidth=3, transform=ax.transAxes)
    ax.text((x0 + x1)/2, y0 + 0.02, f'{length/1000:.0f} km', ha='center',
            va='bottom', transform=ax.transAxes, fontsize=10, fontweight='bold')

# --- CONTOUR MAP ---
def plot_contour(ax, df, title, xlabel, ylabel, unit, levels, cmap, log_scale):
    x, y, z = df['x'], df['y'], df['z']
    xi = np.linspace(x.min(), x.max(), 400)
    yi = np.linspace(y.min(), y.max(), 400)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

    if log_scale:
        Zi = np.log10(Zi + 1e-6)
        levels = np.log10(np.array(levels) + 1e-6)

    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, extend='both')
    cl = ax.contour(Xi, Yi, Zi, levels=levels, colors='black', linewidths=0.6)
    ax.clabel(cl, inline=True, fontsize=8, fmt=lambda v: f"{10**v:.2f}" if log_scale else f"{v:.2f}")

    cbar = plt.colorbar(cf, ax=ax, shrink=0.7)
    label = title
    if unit: label += f" [{unit}]"
    if log_scale: label = f"log₁₀({title} [{unit}])"
    cbar.set_label(label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    add_north_arrow(ax)
    add_scale_bar(ax)

# --- HEATMAP (NET ZERO) ---
def plot_heatmap(ax, df, title, xlabel, ylabel, unit, cmap):
    x, y, z = df['x'], df['y'], df['z']
    xi = np.linspace(x.min(), x.max(), 400)
    yi = np.linspace(y.min(), y.max(), 400)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    Zi = np.ma.masked_invalid(Zi)

    im = ax.imshow(Zi, extent=(xi.min(), xi.max(), yi.min(), yi.max()),
                   origin='lower', cmap=cmap, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{title} [{unit}]" if unit else title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    add_north_arrow(ax)
    add_scale_bar(ax)

# --- HISTOGRAM ---
def plot_histogram(ax, df, title, xlabel, ylabel, bins):
    ax.hist(df['z'], bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

# --- VARIOGRAM HELPERS ---
def compute_variogram(df, max_lag, n_lags, direction=None):
    coords = df[['x', 'y']].values
    z = df['z'].values
    dists = squareform(pdist(coords))
    pairs = np.abs(z[:, None] - z[None, :])

    if direction is not None:
        angle = np.radians(direction)
        vec = np.array([np.cos(angle), np.sin(angle)])
        proj = coords @ vec
        order = np.argsort(proj)
        sorted_z = z[order]
        sorted_proj = proj[order]
        h = np.abs(np.subtract.outer(sorted_proj, sorted_proj))
        dz = np.abs(np.subtract.outer(sorted_z, sorted_z))
        dists = h
        pairs = dz

    lags = np.linspace(0, max_lag, n_lags + 1)
    gamma = []
    centers = []
    for i in range(n_lags):
        mask = (dists >= lags[i]) & (dists < lags[i + 1])
        if mask.sum() > 0:
            gamma.append(0.5 * (pairs[mask] ** 2).mean())
            centers.append((lags[i] + lags[i + 1]) / 2)
    return np.array(centers), np.array(gamma)

# --- NORMAL VARIOGRAM ---
def plot_normal_variogram(ax, df, title, max_lag, n_lags):
    h, gamma = compute_variogram(df, max_lag, n_lags)
    ax.plot(h, gamma, 'o-', color='red')
    ax.set_xlabel('Lag Distance (m)')
    ax.set_ylabel('Semivariance γ(h)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

# --- LONG/SHORT VARIOGRAM ---
def plot_long_short_variogram(ax, df, title, max_lag, n_lags):
    h0, g0 = compute_variogram(df, max_lag, n_lags, direction=0)
    h90, g90 = compute_variogram(df, max_lag, n_lags, direction=90)
    ax.plot(h0, g0, 'o-', label='0° (Long)', color='blue')
    ax.plot(h90, g90, 's-', label='90° (Short)', color='green')
    ax.legend()
    ax.set_xlabel('Lag Distance (m)')
    ax.set_ylabel('Semivariance γ(h)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

# --- UNIFORM LONG VARIOGRAM ---
def plot_uniform_long_variogram(ax, df, title, max_lag, n_lags):
    h, gamma = compute_variogram(df, max_lag, n_lags, direction=0)
    ax.plot(h, gamma, 'o-', color='purple')
    ax.set_xlabel('Lag Distance (m)')
    ax.set_ylabel('Semivariance γ(h)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

# --- OVERLAY ---
def plot_overlay(ax, struct_df, thick_df, title, xlabel, ylabel):
    xi = np.linspace(thick_df['x'].min(), thick_df['x'].max(), 400)
    yi = np.linspace(thick_df['y'].min(), thick_df['y'].max(), 400)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi_thick = griddata((thick_df['x'], thick_df['y']), thick_df['z'], (Xi, Yi), method='cubic')
    cf = ax.contourf(Xi, Yi, Zi_thick, levels=15, cmap="Blues", alpha=0.7)
    plt.colorbar(cf, ax=ax, label="Thickness [m]")

    Zi_struct = griddata((struct_df['x'], struct_df['y']), struct_df['z'], (Xi, Yi), method='linear')
    cs = ax.contour(Xi, Yi, Zi_struct, levels=np.arange(3700, 4100, 20), colors='red', linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%d')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.set_aspect('equal')
    add_north_arrow(ax)
    add_scale_bar(ax)

# ========================================
# UPLOAD
# ========================================
uploaded_files = st.file_uploader("Upload .bna files", type="bna", accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload `.bna` files to begin.")
    st.stop()

all_contours = {f.name: parse_bna(f.read().decode("utf-8")) for f in uploaded_files}

# Session state
if 'config' not in st.session_state:
    st.session_state.config = {}

# ========================================
# SIDEBAR: PER FILE CONFIG
# ========================================
st.sidebar.header("Configure Each File")
for fname in all_contours.keys():
    with st.sidebar.expander(f"{fname}", expanded=False):
        default_title = os.path.splitext(fname)[0].replace("_", " ")
        title = st.text_input("Plot Title", value=default_title, key=f"t_{fname}")

        map_type = st.selectbox("Map Type", MAP_TYPES, key=f"mt_{fname}")

        xlabel = st.text_input("X-axis Label", value="X (m)", key=f"xl_{fname}")
        ylabel = st.text_input("Y-axis Label", value="Y (m)", key=f"yl_{fname}")

        unit = st.text_input("Unit", value="", key=f"u_{fname}")

        # Type-specific settings
        if map_type in ["Contour Map"]:
            z_vals = [c['z'] for c in all_contours[fname]]
            zmin, zmax = min(z_vals), max(z_vals)
            n_levels = st.slider("Contour Levels", 5, 30, 15, key=f"nl_{fname}")
            levels = list(np.linspace(zmin, zmax, n_levels))
            log_scale = st.checkbox("Log Scale", key=f"log_{fname}")
            cmap = st.selectbox("Colormap", ["viridis", "plasma", "terrain", "Blues", "YlOrRd"], key=f"cm_{fname}")
        elif map_type == "Histogram":
            bins = st.slider("Bins", 5, 100, 30, key=f"bins_{fname}")
        elif "Variogram" in map_type:
            max_lag = st.slider("Max Lag (m)", 100, 10000, 3000, step=100, key=f"lag_{fname}")
            n_lags = st.slider("Number of Lags", 10, 50, 20, key=f"nlags_{fname}")

        # Save
        st.session_state.config[fname] = {
            "title": title, "map_type": map_type, "xlabel": xlabel, "ylabel": ylabel,
            "unit": unit, "levels": levels if 'levels' in locals() else None,
            "log_scale": log_scale if 'log_scale' in locals() else False,
            "cmap": cmap if 'cmap' in locals() else "viridis",
            "bins": bins if 'bins' in locals() else 30,
            "max_lag": max_lag if 'max_lag' in locals() else 3000,
            "n_lags": n_lags if 'n_lags' in locals() else 20
        }

# ========================================
# MAIN PLOTTING
# ========================================
plot_mode = st.radio("Plot Mode", ["Single File", "All Files (Batch)"])

if plot_mode == "Single File":
    file = st.selectbox("Select File", options=list(all_contours.keys()))
    cfg = st.session_state.config.get(file, {})
    df = contours_to_df(all_contours[file])

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    mt = cfg.get("map_type", "Contour Map")

    if mt == "Contour Map":
        plot_contour(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"], cfg["unit"],
                     cfg["levels"], cfg["cmap"], cfg["log_scale"])
    elif mt == "Heatmap (Net Zero Contour)":
        plot_heatmap(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"], cfg["unit"], cfg["cmap"])
    elif mt == "Histogram":
        plot_histogram(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"], cfg["bins"])
    elif mt == "Normal Variogram":
        plot_normal_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
    elif mt == "Long/Short Directional Variogram":
        plot_long_short_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
    elif mt == "Uniform Long Variogram":
        plot_uniform_long_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
    elif mt == "Overlay: Structure + Thickness":
        st.warning("Overlay requires two files. Use 'All Files' mode.")
        ax.text(0.5, 0.5, "Select two files in Batch mode", ha='center', transform=ax.transAxes)

    plt.tight_layout()
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button("Download", buf, f"{cfg['title']}.png", "image/png")

else:
    if st.button("Generate All Plots", type="primary"):
        figs = []
        for fname in all_contours.keys():
            cfg = st.session_state.config.get(fname, {})
            df = contours_to_df(all_contours[fname])
            mt = cfg.get("map_type")

            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            if mt == "Contour Map":
                plot_contour(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"], cfg["unit"],
                             cfg["levels"], cfg["cmap"], cfg["log_scale"])
            elif mt == "Heatmap (Net Zero Contour)":
                plot_heatmap(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"], cfg["unit"], cfg["cmap"])
            elif mt == "Histogram":
                plot_histogram(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"], cfg["bins"])
            elif mt == "Normal Variogram":
                plot_normal_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
            elif mt == "Long/Short Directional Variogram":
                plot_long_short_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
            elif mt == "Uniform Long Variogram":
                plot_uniform_long_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
            elif mt == "Overlay: Structure + Thickness":
                continue  # Handle separately

            plt.tight_layout()
            figs.append((cfg["title"], fig))

        # Overlay
        struct_file = thick_file = None
        for f in all_contours.keys():
            if "struct" in f.lower(): struct_file = f
            if "thick" in f.lower(): thick_file = f
        if struct_file and thick_file:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
            plot_overlay(ax, contours_to_df(all_contours[struct_file]),
                         contours_to_df(all_contours[thick_file]),
                         "Overlay: Structure + Thickness", "X (m)", "Y (m)")
            plt.tight_layout()
            figs.append(("Overlay", fig))

        # Display
        for title, fig in figs:
            st.subheader(title)
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            st.download_button(f"Download {title}", buf, f"{title}.png", "image/png")
