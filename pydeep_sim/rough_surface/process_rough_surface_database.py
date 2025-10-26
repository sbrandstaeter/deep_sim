from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

if __name__ == "__main__":

    experiment_name = "rough_surface_points_max_Aeff_0.1_std0_90_L_1000_num_delta_10"
    experiment_name = "rough_surface_points_1"
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

    num_far_field_displacements_per_surface = int((qoi.shape[1] - num_statistics) / 2)
    num_effective_contact_area_fraction_per_surface = (
        num_far_field_displacements_per_surface
    )

    far_field_displacements_names = [
        f"far_field_displacement_{i}"
        for i in range(num_far_field_displacements_per_surface)
    ]

    effective_contact_area_fraction = [
        f"eff_contact_area_fraction_{i}"
        for i in range(num_far_field_displacements_per_surface)
    ]

    hursts = np.repeat(hursts, num_far_field_displacements_per_surface, axis=0)
    num_patches = np.repeat(
        num_patches, num_far_field_displacements_per_surface, axis=0
    )

    surface_statistics = qoi[:, :num_statistics]
    surface_statistics = np.repeat(
        surface_statistics, num_far_field_displacements_per_surface, axis=0
    )

    far_field_displacements = np.ravel(
        qoi[
            :, num_statistics : num_statistics + num_far_field_displacements_per_surface
        ]
    )
    effective_contact_area_fractions = np.ravel(
        qoi[
            :,
            num_statistics
            + num_far_field_displacements_per_surface : num_statistics
            + num_far_field_displacements_per_surface
            + num_effective_contact_area_fraction_per_surface,
        ]
    )

    input_output_df = pd.DataFrame(
        {
            "hurst": hursts,
            "num_patches": num_patches,
            "far_field_displacement": far_field_displacements,
            "effective_contact_area_fraction": effective_contact_area_fractions,
        }
    )

    surface_statistics_df = pd.DataFrame(surface_statistics, columns=statistics_names)

    combined_data_df = pd.concat([input_output_df, surface_statistics_df], axis=1)

    fig = px.scatter_matrix(input_output_df, color="effective_contact_area_fraction")
    fig.show()
