from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

experiment_name = "tamaas_points_nonperiodic_3"

combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")

fig1 = px.scatter(
    combined_data_df,
    x="eff_area",
    y="run_times",
    color="ids",
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Eff area over pressure",  # columns to plot as lines
)
fig1.show()
fig1.write_html(f"{experiment_name}_scatter_eff_area_run_time.html")

fig1 = px.scatter(
    combined_data_df,
    x="dmax",
    y="run_times",
    color="ids",
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={
        "dmax": r"$\Delta$",
        "run_times": "Wall clock time [s]",
    },
)
fig1.show()
fig1.write_html(f"{experiment_name}_scatter_dmax_run_time.html")
