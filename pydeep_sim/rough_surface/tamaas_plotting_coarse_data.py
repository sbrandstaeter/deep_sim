#!/usr/bin/env python3
"""
Load a trained scikit-learn MLP model and evaluate it on a new Parquet test set.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydeep_sim.rough_surface.tamaas_ml_model_utils import RegressionMetrics
from pydeep_sim.rough_surface.figures_generation.plot_style import (
    make_styled_figure,
    MY_BLUE,
)


def main() -> None:
    experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
    base_name = f"{experiment_name}_mlp_scikit_1"
    base_name = f"{experiment_name}_gpr_scikit_1"

    model_path = Path(f"{base_name}_model.joblib")
    new_test_data_path = Path("tamaas_points_nonperiodic_3_coarse_scale.parquet")
    new_test_data_path = Path(
        "tamaas_points_nonperiodic_3_validation_data_coarse.parquet"
    )

    target = "eff_area"

    features = [
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
        "z_max",
        "z_rms",
        "dmax",
    ]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")

    if not new_test_data_path.exists():
        raise FileNotFoundError(
            f"Test data file not found: {new_test_data_path.resolve()}"
        )

    # Load model
    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path.resolve()}")

    # Load test data
    test_df = pd.read_parquet(new_test_data_path)
    print(f"Loaded test data from: {new_test_data_path.resolve()}")
    print(f"Test data shape: {test_df.shape}")

    # Validate columns
    missing_features = [c for c in features if c not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in test data: {missing_features}")

    if target not in test_df.columns:
        raise ValueError(f"Target column '{target}' not found in test data.")

    # Select relevant columns
    test_df = test_df[features + [target]].copy()

    # Check NA
    if test_df.isna().any().any():
        na_counts = test_df.isna().sum()
        raise ValueError(
            "Missing values detected in test data.\n" f"{na_counts[na_counts > 0]}"
        )

    # Check numeric
    non_numeric_features = [
        c for c in features if not pd.api.types.is_numeric_dtype(test_df[c])
    ]
    if non_numeric_features:
        raise TypeError(
            f"Non-numeric feature columns found in test data: {non_numeric_features}"
        )

    # Extract arrays
    X_test = test_df[features].to_numpy()
    y_true = test_df[target].to_numpy()

    # Predict (scaling + inverse transform handled internally)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, None)

    print(f"Prediction done for {len(y_pred)} samples.")

    # --- Use RegressionMetrics ---
    regression_metrics = RegressionMetrics(y_true, y_pred)

    print("#" * 20)
    print("Recommended metrics on coarse data:")
    regression_metrics.print_recommended_table()
    print("#" * 20)

    latex_table = regression_metrics.regression_metrics_to_latex(
        caption="Regression performance on coarse data",
        label="tab:coarse_metrics",
    )
    print("\nLaTeX table:\n")
    print(latex_table)

    # Plot y_pred vs y_true
    fig, ax = make_styled_figure()
    ax.scatter(y_true, y_pred, alpha=0.7)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.grid(True)
    fig.tight_layout()

    out_base = f"tamaas_points_nonperiodic_3_coarse_scale_pred_vs_true"
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".svg")
    fig.savefig(out_base + ".png", dpi=300)

    # dmax vs eff_area ---
    dmax_values = test_df["dmax"].to_numpy()

    fig, ax = make_styled_figure()

    colormap = cm.devon  # "viridis", "turbo", "plasma", "cividis", "inferno", "batlow"

    ax.scatter(
        x,
        y,
        # c=surface_id,
        # cmap=colormap,
        s=9,  # marker size (points^2) → ~3.0 equivalent
        edgecolors="black",
        linewidths=0.3,
        alpha=0.8,
    )

    ax.set_xlabel(r"$\Delta/\bar{\sigma}$")
    ax.set_ylabel(r"$A_\mathrm{e}~(\%)$")

    ax.set_xlim([0.0, 14.0])
    ax.set_ylim([0.0, 45.0])

    ax.set_xticks(np.arange(0, 14.1, 2))
    ax.set_yticks(np.arange(0, 45.1, 5))

    fig.tight_layout()

    out_base = f"tamaas_points_nonperiodic_3_coarse_scale_scatter_dmax_eff_area"
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".svg")
    fig.savefig(out_base + ".png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
