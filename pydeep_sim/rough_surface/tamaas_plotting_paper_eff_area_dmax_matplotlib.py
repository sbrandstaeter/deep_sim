import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------------
# Data
# -------------------------
experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")

# Unused in your plotly snippet (kept here because you load it)
surfaces_mean_std = np.load(f"{experiment_name}_surfaces_stds_mean.npy")

x = combined_data_df["dmax"].to_numpy() / surfaces_mean_std
y = combined_data_df["eff_area"].to_numpy() * 100

# -------------------------
# Matplotlib "print-ready" style to mirror Plotly setup
# -------------------------
latex_font = "Latin Modern Roman"  # like your Plotly choice

# mpl.rcParams["text.usetex"] = True

mpl.rcParams.update(
    {
        # Fonts / text
        "font.family": "serif",
        "font.serif": [
            latex_font,
            "Computer Modern Roman",
            "CMU Serif",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "font.size": 24,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        # Lines / ticks
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        # Grid (light)
        "axes.grid": True,
        "grid.linewidth": 1.0,
        "grid.alpha": 1.0,
        "grid.color": (0, 0, 0, 0.15),
        # Export
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

# -------------------------
# Figure: 900x600 px equivalent
# If you export to vector (PDF/SVG), size is in inches; DPI mainly matters for raster.
# 900/100=9 in, 600/100=6 in
# -------------------------
fig, ax = plt.subplots(figsize=(9, 6))

# Marker style matching Plotly:
# size=9 in plotly is "px-ish"; matplotlib uses points^2 for s.
# 9 px ~ about 6-7 pt visually; tune as needed. We'll pick s=60.
ax.scatter(
    x,
    y,
    s=60,
    alpha=0.85,
    linewidths=0.6,
    edgecolors=(0, 0, 0, 0.35),
)

# Labels (your plotly labels)
ax.set_xlabel(r"${\Delta}/{\bar{\sigma}}$ [-]", labelpad=12)
ax.set_ylabel(r"$A_e$ [%]", labelpad=12)

# Axis ranges/ticks to match your plotly settings
ax.set_xlim(0, 14)
ax.set_ylim(0, 45)

ax.set_xticks(np.arange(0, 14 + 1e-9, 2))
ax.set_yticks(np.arange(0, 45 + 1e-9, 5))

# Mirror spines (Plotly mirror=True)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)

# No zero line (Plotly zeroline=False) -> Matplotlib doesn't add one by default

# Margins similar to Plotly margin=dict(l=80,r=30,t=30,b=70)
# In Matplotlib, use subplot_adjust. These are fractions of figure size:
fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.98)

# Optional: consistent tick placement/formatting
ax.tick_params(which="both", top=True, right=True)

# -------------------------
# Save: vector outputs (publish-ready)
# -------------------------
out_base = f"{experiment_name}_scatter_dmax_eff_area_mpl"
fig.savefig(out_base + ".pdf")  # vector PDF
fig.savefig(out_base + ".svg")  # vector SVG

# Optional raster fallback (high-res)
fig.savefig(out_base + ".png", dpi=300)

plt.show()
