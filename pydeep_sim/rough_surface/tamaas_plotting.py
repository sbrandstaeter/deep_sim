from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

experiment_name = "tamaas_points_nonperiodic_3"

experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"

combined_data_df = pd.read_parquet(f"{experiment_name}.parquet", engine="pyarrow")

fig = px.scatter_matrix(combined_data_df, color="eff_area")
fig.show()
fig.write_html(f"{experiment_name}_scatter_matrix.html")

fig1 = px.scatter(
    combined_data_df,
    x="pressure",
    y="eff_area",
    color="ids",
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Eff area over pressure",  # columns to plot as lines
)
fig1.show()
fig1.write_html(f"{experiment_name}_scatter_pressure_eff_area.html")

fig2 = px.scatter(
    combined_data_df,
    x="dmax",
    y="eff_area",
    color="ids",
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Eff area over dmax",  # columns to plot as lines
)
fig2.show()
fig2.write_html(f"{experiment_name}_scatter_dmax_eff_area.html")

fig3 = px.parallel_coordinates(combined_data_df, color="eff_area")
fig3.show()
fig3.write_html(f"{experiment_name}_parallel_coordinates.html")

fig4 = px.histogram(
    np.unique(combined_data_df["rms_slope"]),
    nbins=20,  # optional
    title="Distribution of rms_slope",
)

fig4.show()
fig4.write_html(f"{experiment_name}_histrogram_rms_slope.html")
