from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


if __name__ == "__main__":

    experiment_name = "rough_surface_points_16"
    output_dir = "./"
    result_file = Path(output_dir) / (experiment_name + ".pickle")

    results = load_result(result_file)

    qoi = results["output"]["result"]
    samples = results["points"]

    parameter_names = ["hurst", "num_patches"]

    hursts = samples["hurst"]
    num_patches = samples["num_patches"]

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
    num_far_field_displacements_per_surface = np.count_nonzero(qoi[0, :] == qoi[0, 0])

    hursts = np.repeat(hursts, num_far_field_displacements_per_surface, axis=0)
    num_patches = np.repeat(
        num_patches, num_far_field_displacements_per_surface, axis=0
    )

    #
    qoi = qoi.ravel().reshape(
        qoi.shape[0] * num_far_field_displacements_per_surface, num_statistics + 4
    )

    surface_statistics = qoi[:, :num_statistics]
    far_field_displacements = qoi[:, num_statistics : num_statistics + 1].ravel()
    effective_contact_area_fractions = qoi[
        :, num_statistics + 1 : num_statistics + 2
    ].ravel()
    pressure = qoi[:, num_statistics + 2 : num_statistics + 3].ravel()
    run_times = qoi[:, num_statistics + 3 : num_statistics + 4].ravel()

    input_output_df = pd.DataFrame(
        {
            "hurst": hursts,
            "num_patches": num_patches,
            "far_field_displacement": far_field_displacements,
            "effective_contact_area_fraction": effective_contact_area_fractions,
            "pressure": pressure,
            "run_times": run_times,
        }
    )

    surface_statistics_df = pd.DataFrame(surface_statistics, columns=statistics_names)

    combined_data_df = pd.concat([input_output_df, surface_statistics_df], axis=1)

    combined_data_df.to_csv(f"{experiment_name}.csv")

    fig = px.scatter_matrix(input_output_df, color="effective_contact_area_fraction")
    fig.show()
    fig.write_html(f"{experiment_name}_scatter_matrix.html")

    fig2 = px.parallel_coordinates(
        input_output_df, color="effective_contact_area_fraction"
    )
    fig2.show()
    fig2.write_html(f"{experiment_name}_parallel_coordinates.html")
