import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures_generation.plot_style import MY_BLUE, make_styled_figure

# -------------------------
# Data
# -------------------------
experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")

# Unused in your plotly snippet (kept here because you load it)
surfaces_mean_std = np.load(f"{experiment_name}_surfaces_stds_mean.npy")

x = combined_data_df["dmax"].to_numpy() / surfaces_mean_std
y = combined_data_df["run_times"].to_numpy()

# -------------------------
# Plot
# -------------------------
fig, ax = make_styled_figure()

ax.plot(
    x,
    y,
    marker="o",
    linestyle="",
    markerfacecolor=MY_BLUE,
    markeredgecolor="black",
    markersize=3.5,
    label="Data",
)

ax.set_xlabel(r"$\Delta/\bar{\sigma}$")
ax.set_ylabel(r"Wall clock time~(s)")

ax.set_xlim([0.0, 14.0])
ax.set_ylim([0.0, 100.0])

ax.set_xticks(np.arange(0, 14.1, 1))
ax.set_yticks(np.arange(0, 100.1, 10))

fig.tight_layout()

# -------------------------
# Save: vector outputs (publish-ready)
# -------------------------
out_base = f"{experiment_name}_scatter_dmax_run_time"
fig.savefig(out_base + ".pdf")  # vector PDF
fig.savefig(out_base + ".svg")  # vector SVG

# Optional raster fallback (high-res)
fig.savefig(out_base + ".png", dpi=300)

plt.show()
