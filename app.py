# -------------------------------------------------
# BNA Reservoir Visualizer – FINAL VERSION
# -------------------------------------------------
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
st.set_page_config(page_title="BNA Reservoir Visualizer", layout="wide")
st.title("BNA Reservoir Contour Analyzer")
st.markdown("Upload any `.bna` files → Choose map type → Download PNG/CSV")

# ========================================
# MAP TYPES
# ========================================
MAP_TYPES = [
    "Structure Map",
    "Thickness Map",
    "Property Heatmap",           # Porosity / Perm / NTG
    "Histogram",
    "Normal Variogram",
    "Long/Short Directional Variogram",
    "Variogram Heatmap",
    "Overlay: Structure + Thickness"
]

DISTRIBUTIONS = ["Normal", "Lognormal", "Uniform"]
VARIOGRAM_RANGES = ["Smooth", "Short", "Long"]

# ========================================
# PARSING .bna
# ========================================
def parse_bna(content: str):
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    contours = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('"C"'):
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) < 3:
                i += 1
                continue
            try:
                z_value = float(parts[1])
                n_points = int(parts[2])
            except Exception:
                i += 1
                continue

            points = []
            for j in range(1, min(n_points + 1, len(lines) - i)):
                coord = lines[i + j]
                if "," in coord:
                    try:
                        x, y = map(float, coord.split(","))
                        points.append((x, y))
                    except Exception:
                        continue
            if len(points) >= 3:
                contours.append({"z": z_value, "points": np.array(points)})
            i += n_points
        i += 1
    return contours

# ========================================
# DATAFRAME FROM CONTOURS
# ========================================
def contours_to_df(contours):
    xs, ys, zs = [], [], []
    for c in contours:
        xs.extend(c["points"][:, 0])
        ys.extend(c["points"][:, 1])
        zs.extend([c["z"]] * len(c["points"]))
    df = pd.DataFrame({"x": xs, "y": ys, "z": zs})
    return df.dropna()

# ========================================
# PLOTTING HELPERS
# ========================================
def add_north_arrow(ax):
    ax.annotate("N", xy=(0.92, 0.92), xycoords="axes fraction",
                ha="center", va="center", fontsize=16, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", edgecolor="black"))
    ax.arrow(0.92, 0.87, 0, 0.04, head_width=0.02, head_length=0.03,
             fc="black", ec="black", transform=ax.transAxes, linewidth=1.5)

def add_scale_bar(ax, length=2000):
    xlim = ax.get_xlim()
    scale = length / (xlim[1] - xlim[0])
    x0, x1 = 0.1, 0.1 + scale
    y0 = 0.05
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=3, transform=ax.transAxes)
    ax.text((x0 + x1)/2, y0 + 0.02, f"{length/1000:.0f} km", ha="center",
            va="bottom", transform=ax.transAxes, fontsize=10, fontweight="bold")

# ---------- STRUCTURE / THICKNESS CONTOUR ----------
def plot_structure_or_thickness(ax, df, title, xlabel, ylabel, unit, levels, cmap):
    x, y, z = df["x"], df["y"], df["z"]
    xi = np.linspace(x.min(), x.max(), 500)
    yi = np.linspace(y.min(), y.max(), 500)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="linear")

    cs = ax.contour(Xi, Yi, Zi, levels=levels, colors='black', linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, alpha=0.7)
    cbar = plt.colorbar(cf, ax=ax, shrink=0.7)
    cbar.set_label(f"{title} [{unit}]" if unit else title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    add_north_arrow(ax)
    add_scale_bar(ax)

# ---------- PROPERTY HEATMAP (WITH SIMULATION TITLE) ----------
def plot_property_heatmap(ax, df, title, xlabel, ylabel, unit, cmap, distribution, vario_range):
    x, y, z = df["x"], df["y"], df["z"]
    xi = np.linspace(x.min(), x.max(), 500)
    yi = np.linspace(y.min(), y.max(), 500)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")
    Zi = np.ma.masked_invalid(Zi)

    im = ax.imshow(Zi, extent=(xi.min(), xi.max(), yi.min(), yi.max()),
                   origin="lower", cmap=cmap, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{title} [{unit}]" if unit else title)

    # Add simulation info to title
    sim_title = f"{title} ({distribution}, {vario_range.lower()} variogram)"
    ax.set_title(sim_title, fontweight="bold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    add_north_arrow(ax)
    add_scale_bar(ax)

# ---------- HISTOGRAM ----------
def plot_histogram(ax, df, title, unit, bins):
    values = df["z"].dropna()
    if values.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return

    counts, edges = np.histogram(values, bins=bins)
    percentages = 100.0 * counts / counts.sum()
    width = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2

    name = title.lower()
    if "porosity" in name:
        color = "#1f77b4"
    elif "permeability" in name or "perm" in name:
        color = "#d62728"
    elif "net" in name or "ntg" in name:
        color = "#2ca02c"
    elif "thickness" in name:
        color = "#ff7f0e"
    else:
        color = "#9467bd"

    ax.bar(centers, percentages, width=width, edgecolor="black", color=color, alpha=0.8)
    ax.set_xlabel(f"{title} [{unit}]" if unit else title)
    ax.set_ylabel("Frequency (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_yticks(np.linspace(0, 100, 6))
    ax.set_yticklabels([f"{int(t)}%" for t in np.linspace(0, 100, 6)])
    ax.grid(True, axis="y", alpha=0.3)

# ---------- VARIOGRAM (LINE) ----------
def compute_variogram(df, max_lag, n_lags, direction=None):
    coords = df[["x", "y"]].values
    z = df["z"].values
    if direction is not None:
        angle = np.radians(direction)
        vec = np.array([np.cos(angle), np.sin(angle)])
        proj = coords @ vec
        order = np.argsort(proj)
        proj, z = proj[order], z[order]
        h = np.abs(np.subtract.outer(proj, proj))
        dz = np.abs(np.subtract.outer(z, z))
    else:
        h = squareform(pdist(coords))
        dz = np.abs(np.subtract.outer(z, z))

    lags = np.linspace(0, max_lag, n_lags + 1)
    gamma, centers = [], []
    for i in range(n_lags):
        mask = (h >= lags[i]) & (h < lags[i + 1])
        if mask.sum() > 0:
            gamma.append(0.5 * (dz[mask] ** 2).mean())
            centers.append((lags[i] + lags[i + 1]) / 2)
    return np.array(centers), np.array(gamma)

def plot_normal_variogram(ax, df, title, max_lag, n_lags):
    h, g = compute_variogram(df, max_lag, n_lags)
    ax.plot(h, g, "o-", color="red")
    ax.set_xlabel("Lag Distance (m)")
    ax.set_ylabel("Semivariance γ(h)")
    ax.set_title(f"{title} - Normal Variogram", fontweight="bold")
    ax.grid(True, alpha=0.3)

def plot_long_short_variogram(ax, df, title, max_lag, n_lags):
    h0, g0 = compute_variogram(df, max_lag, n_lags, direction=0)
    h90, g90 = compute_variogram(df, max_lag, n_lags, direction=90)
    ax.plot(h0, g0, "o-", label="0° (Long)", color="blue")
    ax.plot(h90, g90, "s-", label="90° (Short)", color="green")
    ax.legend()
    ax.set_xlabel("Lag Distance (m)")
    ax.set_ylabel("Semivariance γ(h)")
    ax.set_title(f"{title} - Directional Variogram", fontweight="bold")
    ax.grid(True, alpha=0.3)

# ---------- VARIOGRAM HEATMAP ----------
def plot_variogram_heatmap(ax, df, title, max_lag, n_lags):
    coords = df[["x", "y"]].values
    z = df["z"].values
    h = squareform(pdist(coords))
    dz = np.abs(np.subtract.outer(z, z))
    lag_edges = np.linspace(0, max_lag, n_lags + 1)
    gamma = np.full((n_lags, n_lags), np.nan)
    for i in range(n_lags):
        for j in range(n_lags):
            mask = (h >= lag_edges[i]) & (h < lag_edges[i+1]) & \
                   (h >= lag_edges[j]) & (h < lag_edges[j+1])
            if mask.sum() > 0:
                gamma[i, j] = 0.5 * (dz[mask] ** 2).mean()
    im = ax.imshow(gamma, extent=(0, max_lag, 0, max_lag), origin="lower",
                   cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Semivariance γ(h)")
    ax.set_xlabel("Lag distance (m) – X")
    ax.set_ylabel("Lag distance (m) – Y")
    ax.set_title(f"{title} - Variogram Heatmap", fontweight="bold")

# ---------- OVERLAY ----------
def plot_overlay(ax, struct_df, thick_df):
    xi = np.linspace(thick_df["x"].min(), thick_df["x"].max(), 500)
    yi = np.linspace(thick_df["y"].min(), thick_df["y"].max(), 500)
    Xi, Yi = np.meshgrid(xi, yi)

    # Thickness fill
    Zi_thick = griddata((thick_df["x"], thick_df["y"]), thick_df["z"], (Xi, Yi), method="linear")
    cf = ax.contourf(Xi, Yi, Zi_thick, levels=15, cmap="Blues", alpha=0.7)
    plt.colorbar(cf, ax=ax, label="Thickness [m]")

    # Structure contours
    Zi_struct = griddata((struct_df["x"], struct_df["y"]), struct_df["z"], (Xi, Yi), method="linear")
    cs = ax.contour(Xi, Yi, Zi_struct, levels=np.arange(3700, 4100, 20), colors="red", linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=9, fmt="%d")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Overlay: Structure + Thickness", fontweight="bold")
    ax.set_aspect("equal")
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

# ========================================
# SIDEBAR CONFIG
# ========================================
st.sidebar.header("Map Configuration")
config = {}
for fname in all_contours.keys():
    with st.sidebar.expander(fname, expanded=False):
        default_title = os.path.splitext(fname)[0].replace("_", " ")
        title = st.text_input("Title", value=default_title, key=f"t_{fname}")
        map_type = st.selectbox("Map Type", MAP_TYPES, key=f"mt_{fname}")
        xlabel = st.text_input("X Label", value="X (m)", key=f"xl_{fname}")
        ylabel = st.text_input("Y Label", value="Y (m)", key=f"yl_{fname}")
        unit = st.text_input("Unit", value="", key=f"u_{fname}")

        # Structure / Thickness
        if map_type in ["Structure Map", "Thickness Map"]:
            z_vals = [c["z"] for c in all_contours[fname]]
            zmin, zmax = min(z_vals), max(z_vals)
            n_levels = st.slider("Contour Levels", 5, 50, 20, key=f"nl_{fname}")
            levels = np.linspace(zmin, zmax, n_levels)
            cmap = st.selectbox("Colormap", ["terrain", "Blues", "viridis"], key=f"cm_{fname}")
        else:
            levels = cmap = None

        # Property Heatmap
        if map_type == "Property Heatmap":
            distribution = st.selectbox("Distribution", DISTRIBUTIONS, key=f"dist_{fname}")
            vario_range = st.selectbox("Variogram Range", VARIOGRAM_RANGES, key=f"vr_{fname}")
            cmap = st.selectbox("Colormap", ["YlOrRd", "plasma", "viridis"], key=f"cm_{fname}")
        else:
            distribution = vario_range = None

        # Variogram
        if "Variogram" in map_type:
            max_lag = st.slider("Max Lag (m)", 100, 10000, 3000, step=100, key=f"lag_{fname}")
            n_lags = st.slider("Lags", 10, 50, 20, key=f"nlags_{fname}")
        else:
            max_lag, n_lags = 3000, 20

        # Histogram
        if map_type == "Histogram":
            bins = st.slider("Bins", 5, 100, 30, key=f"bins_{fname}")
        else:
            bins = 30

        config[fname] = {
            "title": title,
            "map_type": map_type,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "unit": unit,
            "levels": levels,
            "cmap": cmap,
            "distribution": distribution,
            "vario_range": vario_range,
            "max_lag": max_lag,
            "n_lags": n_lags,
            "bins": bins,
        }

# ========================================
# PLOTTING
# ========================================
plot_mode = st.radio("Plot Mode", ["Single File", "All Files (Batch)"])

if plot_mode == "Single File":
    file = st.selectbox("Select File", options=list(all_contours.keys()))
    cfg = config[file]
    df = contours_to_df(all_contours[file])
    fig, ax = plt.subplots(figsize=(11, 9), dpi=160)

    mt = cfg["map_type"]
    if mt == "Structure Map":
        plot_structure_or_thickness(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"],
                                    cfg["unit"], cfg["levels"], cfg["cmap"])
    elif mt == "Thickness Map":
        plot_structure_or_thickness(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"],
                                    cfg["unit"], cfg["levels"], cfg["cmap"])
    elif mt == "Property Heatmap":
        plot_property_heatmap(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"],
                              cfg["unit"], cfg["cmap"], cfg["distribution"], cfg["vario_range"])
    elif mt == "Histogram":
        plot_histogram(ax, df, cfg["title"], cfg["unit"], cfg["bins"])
    elif mt == "Normal Variogram":
        plot_normal_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
    elif mt == "Long/Short Directional Variogram":
        plot_long_short_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
    elif mt == "Variogram Heatmap":
        plot_variogram_heatmap(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
    elif mt == "Overlay: Structure + Thickness":
        s_file = next((f for f in all_contours if "struct" in f.lower()), None)
        t_file = next((f for f in all_contours if "thick" in f.lower()), None)
        if s_file and t_file:
            plot_overlay(ax, contours_to_df(all_contours[s_file]), contours_to_df(all_contours[t_file]))
        else:
            st.error("Need both Structure and Thickness files.")
    plt.tight_layout()
    st.pyplot(fig)

    # Export
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    st.download_button("Download PNG", buf, f"{cfg['title']}.png", "image/png")

    if st.checkbox("Export Grid CSV"):
        xi = np.linspace(df["x"].min(), df["x"].max(), 500)
        yi = np.linspace(df["y"].min(), df["y"].max(), 500)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((df["x"], df["y"]), df["z"], (Xi, Yi), method="cubic")
        grid_df = pd.DataFrame({"X": Xi.ravel(), "Y": Yi.ravel(), "Z": Zi.ravel()})
        csv = grid_df.to_csv(index=False)
        st.download_button("Download Grid CSV", csv, f"{cfg['title']}_grid.csv", "text/csv")

else:  # Batch
    if st.button("Generate All Plots", type="primary"):
        figs = []
        for fname in all_contours.keys():
            cfg = config[fname]
            df = contours_to_df(all_contours[fname])
            fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
            mt = cfg["map_type"]

            if mt == "Structure Map":
                plot_structure_or_thickness(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"],
                                            cfg["unit"], cfg["levels"], cfg["cmap"])
            elif mt == "Thickness Map":
                plot_structure_or_thickness(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"],
                                            cfg["unit"], cfg["levels"], cfg["cmap"])
            elif mt == "Property Heatmap":
                plot_property_heatmap(ax, df, cfg["title"], cfg["xlabel"], cfg["ylabel"],
                                      cfg["unit"], cfg["cmap"], cfg["distribution"], cfg["vario_range"])
            elif mt == "Histogram":
                plot_histogram(ax, df, cfg["title"], cfg["unit"], cfg["bins"])
            elif mt == "Normal Variogram":
                plot_normal_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
            elif mt == "Long/Short Directional Variogram":
                plot_long_short_variogram(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
            elif mt == "Variogram Heatmap":
                plot_variogram_heatmap(ax, df, cfg["title"], cfg["max_lag"], cfg["n_lags"])
            plt.tight_layout()
            figs.append((cfg["title"], fig))

        # Overlay
        s_file = next((f for f in all_contours if "struct" in f.lower()), None)
        t_file = next((f for f in all_contours if "thick" in f.lower()), None)
        if s_file and t_file:
            fig, ax = plt.subplots(figsize=(12, 9), dpi=160)
            plot_overlay(ax, contours_to_df(all_contours[s_file]), contours_to_df(all_contours[t_file]))
            plt.tight_layout()
            figs.append(("Overlay: Structure + Thickness", fig))

        for title, fig in figs:
            st.subheader(title)
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
            buf.seek(0)
            st.download_button(f"Download {title}.png", buf, f"{title}.png", "image/png")
