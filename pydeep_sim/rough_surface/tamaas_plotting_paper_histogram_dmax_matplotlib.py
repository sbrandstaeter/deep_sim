import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures_generation.plot_style import make_styled_figure, MY_BLUE

# -------------------------
# Data
# -------------------------
experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
# experiment_name = "tamaas_points_nonperiodic_3_test_data"

combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")
surfaces_mean_std = np.load(f"{experiment_name}_surfaces_stds_mean.npy")

x = combined_data_df["dmax"].to_numpy() / surfaces_mean_std

# -------------------------
# Plot
# -------------------------
fig, ax = make_styled_figure()

bins = np.linspace(0, 14, 29)

ax.hist(
    x,
    bins=bins,
    density=False,
    color=MY_BLUE,
    edgecolor="white",
    linewidth=0.5,
)

ax.set_xlabel(r"$\Delta/\bar{\sigma}$")
ax.set_ylabel(r"Number of samples")

ax.set_xlim([0.0, 14.0])
ax.set_xticks(np.arange(0, 14.1, 2))

fig.tight_layout()

# -------------------------
# Save
# -------------------------
out_base = f"{experiment_name}_hist_dmax"
fig.savefig(out_base + ".pdf")
fig.savefig(out_base + ".svg")
fig.savefig(out_base + ".png", dpi=300)

plt.show()
