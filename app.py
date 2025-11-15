import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import os
from matplotlib.colors import LogNorm
import io

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(page_title="BNA Contour Visualizer", layout="wide")
st.title("Dynamic BNA Contour Map Generator")
st.markdown("Upload `.bna` files → Assign **custom name, units, levels, colormap** → Generate **pro maps**")

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
# PLOTTING UTILS
# ========================================
def add_north_arrow(ax, x=0.92, y=0.92):
    ax.annotate('N', xy=(x, y), xycoords='axes fraction',
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="circle,pad=0.3", facecolor="white", edgecolor="black"))
    ax.arrow(x, y - 0.05, 0, 0.04, head_width=0.02, head_length=0.03,
             fc='black', ec='black', transform=ax.transAxes, linewidth=1.5)

def add_scale_bar(ax, length=2000, loc=(0.1, 0.05)):
    xlim = ax.get_xlim()
    scale = length / (xlim[1] - xlim[0])
    x0, x1 = loc[0], loc[0] + scale
    y0 = loc[1]
    ax.plot([x0, x1], [y0, y0], color='black', linewidth=3, transform=ax.transAxes)
    ax.text((x0 + x1)/2, y0 + 0.02, f'{length/1000:.0f} km', ha='center',
            va='bottom', transform=ax.transAxes, fontsize=10, fontweight='bold')

def plot_contour_map(ax, df, title, unit, levels, cmap, log_scale, grid_density=400):
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    xi = np.linspace(x.min(), x.max(), grid_density)
    yi = np.linspace(y.min(), y.max(), grid_density)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

    if log_scale:
        Zi = np.log10(Zi + 1e-6)
        levels = np.log10(np.array(levels) + 1e-6)

    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, extend='both')
    cl = ax.contour(Xi, Yi, Zi, levels=levels, colors='black', linewidths=0.6, alpha=0.8)
    ax.clabel(cl, inline=True, fontsize=8, fmt=lambda v: f"{10**v:.2f}" if log_scale else f"{v:.2f}")

    cbar = plt.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    label = title
    if unit:
        label += f" [{unit}]"
    if log_scale:
        label = f"log₁₀({title} [{unit}])"
    cbar.set_label(label, fontsize=10)

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title(title, fontsize=12, pad=15, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    add_north_arrow(ax)
    add_scale_bar(ax)

# ========================================
# UPLOAD & STATE
# ========================================
uploaded_files = st.file_uploader(
    "Upload .bna files", type="bna", accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload one or more `.bna` files to start.")
    st.stop()

# Parse all
all_contours = {}
for file in uploaded_files:
    content = file.read().decode("utf-8")
    all_contours[file.name] = parse_bna(content)

# Store user settings in session state
if 'user_config' not in st.session_state:
    st.session_state.user_config = {}

# ========================================
# SIDEBAR: USER CONFIG PER FILE
# ========================================
st.sidebar.header("Configure Each File")

for fname in all_contours.keys():
    with st.sidebar.expander(f"⚙️ {fname}", expanded=False):
        default_title = os.path.splitext(fname)[0].replace("_", " ")
        title = st.text_input("Map Title", value=default_title, key=f"title_{fname}")

        unit = st.text_input("Unit (e.g. m, md, fraction)", value="", key=f"unit_{fname}")

        # Auto-detect range
        all_z = [c['z'] for c in all_contours[fname]]
        z_min, z_max = min(all_z), max(all_z)
        st.write(f"Z range: {z_min:.3f} → {z_max:.3f}")

        log_scale = st.checkbox("Log scale?", key=f"log_{fname}")

        if log_scale:
            levels_input = st.text_input(
                "Levels (comma-separated, e.g. 1,10,50,100,500)",
                value="1,10,50,100,500", key=f"levels_{fname}"
            )
        else:
            n_levels = st.slider("Number of contour levels", 5, 30, 15, key=f"nlev_{fname}")
            levels_input = ",".join(map(str, np.linspace(z_min, z_max, n_levels)))

        try:
            levels = [float(x) for x in levels_input.split(',') if x.strip()]
        except:
            levels = np.linspace(z_min, z_max, 10)
            st.warning("Invalid levels → using auto")

        cmap = st.selectbox("Colormap", [
            "viridis", "plasma", "inferno", "magma", "cividis",
            "terrain", "Blues", "Greens", "YlOrRd", "hot"
        ], key=f"cmap_{fname}")

        # Save to session
        st.session_state.user_config[fname] = {
            "title": title,
            "unit": unit,
            "levels": levels,
            "cmap": cmap,
            "log_scale": log_scale
        }

# ========================================
# MAIN: PLOT OPTIONS
# ========================================
plot_option = st.radio("Plot Mode", ["Single File", "Generate All Maps"])

# --- SINGLE FILE ---
if plot_option == "Single File":
    selected_file = st.selectbox("Select file", options=list(all_contours.keys()))
    config = st.session_state.user_config.get(selected_file, {})

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    # Build df
    all_x, all_y, all_z = [], [], []
    for c in all_contours[selected_file]:
        xs, ys = c['points'][:, 0], c['points'][:, 1]
        all_x.extend(xs); all_y.extend(ys); all_z.extend([c['z']] * len(xs))
    df = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})

    plot_contour_map(
        ax=ax,
        df=df,
        title=config.get("title", selected_file),
        unit=config.get("unit", ""),
        levels=config.get("levels", []),
        cmap=config.get("cmap", "viridis"),
        log_scale=config.get("log_scale", False)
    )
    plt.tight_layout()
    st.pyplot(fig)

    # Download
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button("Download Map", buf, f"{config['title']}.png", "image/png")

# --- GENERATE ALL MAPS ---
else:
    if st.button("Generate All Maps", type="primary"):
        n_files = len(all_contours)
        cols = 2
        rows = (n_files + 1) // 2 + 1  # +1 for overlay

        fig = plt.figure(figsize=(16, 8 * rows))
        gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.3)

        plot_idx = 0
        for fname in all_contours.keys():
            if plot_idx >= rows * cols:
                break
            ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
            config = st.session_state.user_config.get(fname, {})

            all_x, all_y, all_z = [], [], []
            for c in all_contours[fname]:
                xs, ys = c['points'][:, 0], c['points'][:, 1]
                all_x.extend(xs); all_y.extend(ys); all_z.extend([c['z']] * len(xs))
            df = pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})

            plot_contour_map(
                ax=ax,
                df=df,
                title=config.get("title", fname),
                unit=config.get("unit", ""),
                levels=config.get("levels", []),
                cmap=config.get("cmap", "viridis"),
                log_scale=config.get("log_scale", False)
            )
            plot_idx += 1

        # Overlay (last row, full width)
        if len(all_contours) >= 2:
            ax = fig.add_subplot(gs[-1, :])
            files = list(all_contours.keys())
            struct_df = thick_df = None
            for f in files:
                if "struct" in f.lower() or "top" in f.lower():
                    struct_df = get_df(all_contours[f])
                if "thick" in f.lower():
                    thick_df = get_df(all_contours[f])
            if struct_df is not None and thick_df is not None:
                plot_overlay(ax, struct_df, thick_df)
            else:
                ax.text(0.5, 0.5, "Need Structure + Thickness files for overlay", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_xticks([]); ax.set_yticks([])

        st.pyplot(fig)

        # Download all
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.download_button("Download All Maps", buf, "all_reservoir_maps.png", "image/png")

# Helper
def get_df(contours):
    all_x, all_y, all_z = [], [], []
    for c in contours:
        xs, ys = c['points'][:, 0], c['points'][:, 1]
        all_x.extend(xs); all_y.extend(ys); all_z.extend([c['z']] * len(xs))
    return pd.DataFrame({'x': all_x, 'y': all_y, 'z': all_z})

def plot_overlay(ax, struct_df, thick_df):
    xi = np.linspace(thick_df['x'].min(), thick_df['x'].max(), 400)
    yi = np.linspace(thick_df['y'].min(), thick_df['y'].max(), 400)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi_thick = griddata((thick_df['x'], thick_df['y']), thick_df['z'], (Xi, Yi), method='cubic')
    cf = ax.contourf(Xi, Yi, Zi_thick, levels=15, cmap="Blues", alpha=0.7)
    plt.colorbar(cf, ax=ax, label="Thickness [m]")

    Zi_struct = griddata((struct_df['x'], struct_df['y']), struct_df['z'], (Xi, Yi), method='linear')
    cs = ax.contour(Xi, Yi, Zi_struct, levels=np.arange(3700, 4100, 20), colors='red', linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=9, fmt='%d')
    ax.set_title("Overlay: Structure (red) + Thickness", fontweight='bold')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    add_north_arrow(ax); add_scale_bar(ax)
