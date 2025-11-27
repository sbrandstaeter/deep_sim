from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def _exp_model(x, a, b, c):
    return a * np.exp(b * x) + c


def _fit_exponential(x, y):
    # Robust initial guesses
    c0 = float(np.median(y))
    a0 = float(np.max(y) - c0) if np.max(y) != c0 else 1.0
    valid = y - c0 > 0
    if np.sum(valid) >= 2:
        b0 = float(np.polyfit(x[valid], np.log(y[valid] - c0), 1)[0])
    else:
        b0 = 0.0
    p0 = [a0, b0, c0]
    popt, pcov = curve_fit(_exp_model, x, y, p0=p0, maxfev=10000)
    return popt, pcov


def _fit_cubic(x, y):
    coeffs = np.polyfit(x, y, 3)  # [p3, p2, p1, p0]
    return coeffs, np.poly1d(coeffs)


def fit_and_plot(
    x,
    y,
    out_path="fit_plot.png",
    title="Curve Fitting: Exponential vs Cubic",
    n_points=1000,
):
    """
    x, y: 1D numpy arrays of equal length
    out_path: output image filename
    Returns dict with parameters and RMSEs.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    # Sort by x for clean plotting
    order = np.argsort(x)
    x, y = x[order], y[order]

    # Fits
    exp_params, exp_covar = _fit_exponential(x, y)  # [a, b, c]
    cubic_coeffs, cubic_fn = _fit_cubic(x, y)  # [p3, p2, p1, p0]

    # Smooth grid
    xs = np.linspace(x.min(), x.max(), int(n_points))
    y_exp = _exp_model(xs, *exp_params)
    y_cubic = cubic_fn(xs)

    # RMSE
    rmse_exp = float(np.sqrt(np.mean((_exp_model(x, *exp_params) - y) ** 2)))
    rmse_cubic = float(np.sqrt(np.mean((cubic_fn(x) - y) ** 2)))

    # Plot (one figure, all curves)
    plt.figure(figsize=(9, 6), dpi=120)
    plt.scatter(x, y, s=25, label="Data", alpha=0.85)
    plt.plot(
        xs,
        y_exp,
        linewidth=2,
        label=f"Exponential: a={exp_params[0]:.4g}, b={exp_params[1]:.4g}, c={exp_params[2]:.4g} (RMSE={rmse_exp:.3g})",
    )
    plt.plot(
        xs,
        y_cubic,
        linewidth=2,
        label=f"Cubic: {cubic_coeffs[0]:.4g}x³ + {cubic_coeffs[1]:.4g}x² + {cubic_coeffs[2]:.4g}x + {cubic_coeffs[3]:.4g} (RMSE={rmse_cubic:.3g})",
    )
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return {
        "exp_params": {"a": exp_params[0], "b": exp_params[1], "c": exp_params[2]},
        "cubic_coeffs": {
            "x^3": cubic_coeffs[0],
            "x^2": cubic_coeffs[1],
            "x": cubic_coeffs[2],
            "const": cubic_coeffs[3],
        },
        "rmse": {"exponential": rmse_exp, "cubic": rmse_cubic},
        "out_path": out_path,
    }


if __name__ == "__main__":

    experiment_name = "rough_surface_points_max_Aeff_0.1_std0_90_L_1000_num_delta_10"
    experiment_name = "rough_surface_points_10"
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

    # reverse engineer repeat operation
    num_far_field_displacements_per_surface = np.count_nonzero(qoi[0, :] == qoi[0, 0])
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

    #
    qoi = qoi.ravel().reshape(
        qoi.shape[0] * num_far_field_displacements_per_surface, num_statistics + 3
    )

    surface_statistics = qoi[:, :num_statistics]
    far_field_displacements = qoi[:, num_statistics : num_statistics + 1].ravel()
    effective_contact_area_fractions = qoi[
        :, num_statistics + 1 : num_statistics + 2
    ].ravel()
    run_times = qoi[:, num_statistics + 2 : num_statistics + 3].ravel()

    input_output_df = pd.DataFrame(
        {
            "hurst": hursts,
            "num_patches": num_patches,
            "far_field_displacement": far_field_displacements,
            "effective_contact_area_fraction": effective_contact_area_fractions,
            "run_times": run_times,
        }
    )

    surface_statistics_df = pd.DataFrame(surface_statistics, columns=statistics_names)

    combined_data_df = pd.concat([input_output_df, surface_statistics_df], axis=1)

    combined_data_df.to_csv(f"{experiment_name}.csv")

    fig = px.scatter_matrix(input_output_df, color="effective_contact_area_fraction")
    fig.show()
    fig.write_html(f"{experiment_name}_scatter_matrix.html")

    results = fit_and_plot(
        effective_contact_area_fractions,
        run_times,
        out_path=f"{experiment_name}_curve_fit.png",
    )
    print(results)
