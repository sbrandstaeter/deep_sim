import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures_generation.plot_style import make_styled_figure, MY_BLUE
from cmcrameri import cm

# -------------------------
# Data
# -------------------------
experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
# experiment_name = "tamaas_points_nonperiodic_3_test_data"

combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")
surfaces_mean_std = np.load(f"{experiment_name}_surfaces_stds_mean.npy")

x = combined_data_df["dmax"].to_numpy() / surfaces_mean_std
y = combined_data_df["eff_area"].to_numpy() * 100
surface_id = combined_data_df["ids"].to_numpy()

# -------------------------
# Plot
# -------------------------
fig, ax = make_styled_figure()

colormap = cm.devon  # "viridis", "turbo", "plasma", "cividis", "inferno", "batlow"

ax.scatter(
    x,
    y,
    # c=surface_id,
    # cmap=colormap,
    s=9,  # marker size (points^2) → ~3.0 equivalent
    edgecolors="black",
    linewidths=0.3,
    alpha=0.8,
)


ax.set_xlabel(r"$\Delta/\bar{\sigma}$")
ax.set_ylabel(r"$A_\mathrm{e}~(\%)$")

ax.set_xlim([0.0, 14.0])
ax.set_ylim([0.0, 45.0])

ax.set_xticks(np.arange(0, 14.1, 2))
ax.set_yticks(np.arange(0, 45.1, 5))

fig.tight_layout()

out_base = f"{experiment_name}_scatter_dmax_eff_area"
fig.savefig(out_base + ".pdf")
fig.savefig(out_base + ".svg")
fig.savefig(out_base + ".png", dpi=300)

plt.show()
