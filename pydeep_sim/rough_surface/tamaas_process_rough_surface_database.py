from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


if __name__ == "__main__":

    experiment_name = "tamaas_points_4"
    output_dir = "./"
    result_file = Path(output_dir) / (experiment_name + ".pickle")

    results = load_result(result_file)

    qoi = results["output"]["result"]
    samples = results["points"]

    parameter_names = ["hurst", "num_patches"]

    hursts = samples["hurst"]
    q1s = samples["q1"]
    q2s = samples["q2"]
    random_seeds = samples["random_seed"]

    statistics_names = [
        "mean_z_peaks",
        "rms_z_peaks",
        "rms_slope_peaks",
        "ks_z_peaks",
        "sk_z_peaks",
        "mean_curv_peaks",
        "ks_curv_peaks",
        "sk_curv_peaks",
        "dn_peaks",
        "alfa_x",
        "alfa_y",
        "mean_z_asp",
        "rms_z_asp",
        "ks_z_asp",
        "sk_z_asp",
        "mean_curv_asp",
        "rms_curv_asp",
        "ks_curv_asp",
        "sk_curv_asp",
        "dns_asp",
        "z_mean",
        "z_max",
        "z_rms",
    ]
    num_statistics = len(statistics_names)

    # reverse engineer repeat operation
    num_pressure_steps_per_surface = np.count_nonzero(qoi[0, :] == qoi[0, 0])

    hursts = np.repeat(hursts, num_pressure_steps_per_surface, axis=0)
    q1s = np.repeat(q1s, num_pressure_steps_per_surface, axis=0)
    q2s = np.repeat(q2s, num_pressure_steps_per_surface, axis=0)
    random_seeds = np.repeat(random_seeds, num_pressure_steps_per_surface, axis=0)

    #
    qoi = qoi.ravel().reshape(
        qoi.shape[0] * num_pressure_steps_per_surface, num_statistics + 6
    )

    surface_statistics = qoi[:, :num_statistics]
    pressure = qoi[:, num_statistics : num_statistics + 1].ravel()
    effective_contact_area_fractions = qoi[
        :, num_statistics + 1 : num_statistics + 2
    ].ravel()
    dmean = qoi[:, num_statistics + 2 : num_statistics + 3].ravel()
    dmin = qoi[:, num_statistics + 3 : num_statistics + 4].ravel()
    dmax = qoi[:, num_statistics + 4 : num_statistics + 5].ravel()
    run_times = qoi[:, num_statistics + 5 : num_statistics + 6].ravel()

    input_output_df = pd.DataFrame(
        {
            "hurst": hursts,
            "q1": q1s,
            "q2": q2s,
            "dmax": dmax,
            "eff_area": effective_contact_area_fractions,
            "pressure": pressure,
            "run_times": run_times,
        }
    )

    surface_statistics_df = pd.DataFrame(surface_statistics, columns=statistics_names)

    combined_data_df = pd.concat([input_output_df, surface_statistics_df], axis=1)

    combined_data_df.to_csv(f"{experiment_name}.csv")

    fig = px.scatter_matrix(input_output_df, color="eff_area")
    fig.show()
    fig.write_html(f"{experiment_name}_scatter_matrix.html")

    fig = px.scatter(
        input_output_df,
        x="pressure",
        y="eff_area",
        title="Eff area over pressure",  # columns to plot as lines
    )
    fig.show()
    fig.write_html(f"{experiment_name}_scatter_pressure_eff_area.html")

    fig = px.scatter(
        input_output_df,
        x="dmax",
        y="eff_area",
        title="Eff area over dmax",  # columns to plot as lines
    )
    fig.show()
    fig.write_html(f"{experiment_name}_scatter_dmax_eff_area.html")

    fig2 = px.parallel_coordinates(input_output_df, color="eff_area")
    fig2.show()
    fig2.write_html(f"{experiment_name}_parallel_coordinates.html")
