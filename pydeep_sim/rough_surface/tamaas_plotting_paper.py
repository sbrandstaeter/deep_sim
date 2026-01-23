import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

experiment_name = "tamaas_points_nonperiodic_3"

combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")

surfaces_mean_std = np.load(f"{experiment_name}_surfaces_stds_mean.npy")

# Get unique categories
ids = combined_data_df["ids"].unique()
colors = cm.Set2.colors  # same palette as Plotly Set2

plt.figure(figsize=(8, 6))

for i, id_val in enumerate(ids):
    subset = combined_data_df[combined_data_df["ids"] == id_val]
    plt.scatter(
        subset["dmax"],
        subset["run_times"],
        label=id_val,
        color=colors[i % len(colors)],
        alpha=0.8,
    )

plt.xlabel(r"$\Delta$")
plt.ylabel("Wall clock time [s]")
plt.legend(title="ids")
plt.tight_layout()

plt.show()
plt.savefig(f"{experiment_name}_scatter_dmax_run_time.png", dpi=300)
