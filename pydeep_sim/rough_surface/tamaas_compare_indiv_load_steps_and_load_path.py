from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

experiment_name = "tamaas_points_nonperiodic_3"
combined_data_load_path_df = pd.read_parquet(
    f"{experiment_name}.parquet", engine="pyarrow"
)

experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
combined_data_indiv_load_steps_df = pd.read_parquet(
    f"{experiment_name}.parquet", engine="pyarrow"
)

col_to_ignore = "run_times"
from pandas.testing import assert_frame_equal

assert_frame_equal(
    combined_data_load_path_df.drop(columns=[col_to_ignore]),
    combined_data_indiv_load_steps_df.drop(columns=[col_to_ignore]),
)
assert_frame_equal(
    combined_data_load_path_df.drop(columns=[col_to_ignore]),
    combined_data_indiv_load_steps_df.drop(columns=[col_to_ignore]),
    rtol=1e-7,
    atol=1e-12,
)
assert_frame_equal(
    combined_data_load_path_df,
    combined_data_indiv_load_steps_df,
)
