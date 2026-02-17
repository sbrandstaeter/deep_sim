from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


if __name__ == "__main__":

    experiment_name = "tamaas_points_nonperiodic_1"
    experiment_name = "tamaas_points_surfaces_not_scaled"
    experiment_name = "tamaas_points_surfaces_scaled"
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
    num_param_combinations = len(q1s)

    statistics_names = [
        "mean_z_peaks",
        "rms_z_peaks",
        "rms_slope",
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
    ids = np.repeat(
        np.arange(0, num_param_combinations),
        num_pressure_steps_per_surface,
        axis=0,
    )
    q1s = np.repeat(q1s, num_pressure_steps_per_surface, axis=0)
    q2s = np.repeat(q2s, num_pressure_steps_per_surface, axis=0)
    random_seeds = np.repeat(random_seeds, num_pressure_steps_per_surface, axis=0)

    #
    qoi = qoi.ravel().reshape(
        qoi.shape[0] * num_pressure_steps_per_surface, num_statistics + 6
    )

    surface_statistics = qoi[:, :num_statistics]
    # mean_rms_slope = np.mean(surface_statistics[:, 2])
    # surface_statistics[:, 2] /= mean_rms_slope
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
            "ids": ids,
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

    fig = px.scatter(
        combined_data_df,
        x="q1",
        y="rms_slope",
        # color="ids",
        # color_discrete_sequence=px.colors.qualitative.Set2,
        title="RMS slope over q1",  # columns to plot as lines
    )
    fig.show()
    fig.write_html(f"{experiment_name}_scatter_q1_rms_slope.html")

    fig2 = px.parallel_coordinates(
        combined_data_df,
        dimensions=["hurst", "q1", "q2", "rms_slope"],
        color="rms_slope",
    )
    fig2.show()
    fig2.write_html(f"{experiment_name}_parallel_coordinates_rms_slope.html")

    fig4 = px.histogram(
        np.unique(combined_data_df["rms_slope"]),
        nbins=20,  # optional
        title="Distribution of rms_slope",
    )

    fig4.show()
    fig4.write_html(f"{experiment_name}_histrogram_rms_slope.html")

    mean_rms_slopes = []

    for q1 in [1, 4, 16]:
        for q2 in [32, 64, 128]:
            mean_rms_slopes.append(
                np.mean(
                    combined_data_df.loc[
                        (combined_data_df["q1"] == q1) & (combined_data_df["q2"] == q2),
                        "rms_slope",
                    ]
                )
            )

    print(mean_rms_slopes)
