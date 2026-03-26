import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures_generation.plot_style import make_styled_figure, MY_BLUE

# -------------------------
# Data
# -------------------------
experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"

combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")

x = combined_data_df["pressure"].to_numpy()

# -------------------------
# Plot
# -------------------------
fig, ax = make_styled_figure()

bins = np.linspace(0, 0.4, 51)

ax.hist(
    x,
    bins=bins,
    density=False,
    color=MY_BLUE,
    edgecolor="white",
    linewidth=0.5,
)

# Labels
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"Number of samples")

# Axis settings
ax.set_xlim([0.0, 0.4])
ax.set_xticks(np.arange(0, 0.401, 0.05))

fig.tight_layout()

# -------------------------
# Save
# -------------------------
out_base = f"{experiment_name}_hist_pressure"
fig.savefig(out_base + ".pdf")
fig.savefig(out_base + ".svg")
fig.savefig(out_base + ".png", dpi=300)

plt.show()
