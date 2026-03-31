#!/usr/bin/env python3
"""
Load a trained GPflow GPR model from checkpoint and evaluate it on a
1D sweep in dmax while keeping all other statistical parameters fixed.

Behavior:
- Rebuilds the same GPflow GPR architecture used in training
- Loads the fitted StandardScaler
- Restores the GPflow model from TensorFlow checkpoint
- Sets all features except dmax to fixed values taken as training-set medians
- Evaluates predictions for dmax in linspace(0, 1, 11), excluding 0
- Plots predicted eff_area vs far-field displacement / dmax

Notes:
- The model was trained on log1p(target), so predictions are inverse-transformed
  with expm1 back to the original target scale.
- This script assumes the same feature list and checkpoint/scaler naming scheme
  as in your training script.
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow


def configure_tensorflow_device(device_preference: str = "auto") -> str:
    """
    Configure TensorFlow device placement.

    Args:
        device_preference:
            - "auto": use GPU if available, else CPU
            - "gpu": require GPU
            - "cpu": force CPU

    Returns:
        TensorFlow device string, e.g. "/GPU:0" or "/CPU:0"
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if device_preference == "cpu":
        print("Forcing CPU execution.")
        return "/CPU:0"

    if device_preference == "gpu":
        if not gpus:
            raise RuntimeError(
                "GPU was requested, but TensorFlow does not see any GPU devices."
            )
        print(f"Using GPU: {gpus[0].name}")
        return "/GPU:0"

    if gpus:
        print(f"Using GPU: {gpus[0].name}")
        return "/GPU:0"

    print("No GPU detected by TensorFlow. Falling back to CPU.")
    if not cpus:
        raise RuntimeError("TensorFlow does not report any CPU devices either.")
    return "/CPU:0"


def build_gpflow_gpr(
    X_reference_scaled: np.ndarray,
    y_reference_log: np.ndarray,
    num_features: int,
) -> gpflow.models.GPR:
    """
    Build the same GPflow exact GPR model as in training.

    GPflow expects:
      X: shape [N, D]
      Y: shape [N, P]

    We supply reference arrays only to instantiate the model structure before
    checkpoint restore. The restored checkpoint will overwrite parameter values.
    """
    X_tf = X_reference_scaled.astype(np.float64)
    y_tf = y_reference_log.reshape(-1, 1).astype(np.float64)

    kernel = gpflow.kernels.SquaredExponential(
        lengthscales=np.ones(num_features, dtype=np.float64),
        variance=1.0,
    )

    model = gpflow.models.GPR(
        data=(X_tf, y_tf),
        kernel=kernel,
        mean_function=None,
        noise_variance=1e-5,
    )

    gpflow.set_trainable(model.likelihood.variance, False)
    return model


def main() -> None:
    gpflow.config.set_default_float(np.float64)

    device_preference = "gpu"  # "auto", "gpu", or "cpu"
    device_name = configure_tensorflow_device(device_preference)

    experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
    base_name = f"{experiment_name}_gpr_gpflow_2"

    checkpoint_dir = Path(f"{base_name}_ckpt")
    scaler_path = Path(f"{base_name}_scaler.joblib")
    train_data_path = Path(f"{experiment_name}.parquet")

    output_csv = Path(f"{base_name}_dmax_sweep_predictions.csv")
    output_png = Path(f"{base_name}_dmax_sweep_plot.png")

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

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path.resolve()}")

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir.resolve()}"
        )

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint found in directory: {checkpoint_dir.resolve()}"
        )

    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Training parquet not found: {train_data_path.resolve()}"
        )

    train_df_raw = pd.read_parquet(train_data_path)

    missing_features = [c for c in features if c not in train_df_raw.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    if target not in train_df_raw.columns:
        raise ValueError(f"Missing target column: {target}")

    df = train_df_raw[features + [target]].copy()

    if df.isna().any().any():
        na_counts = df.isna().sum()
        raise ValueError(
            "Missing values detected in training data.\n" f"{na_counts[na_counts > 0]}"
        )

    non_numeric_features = [
        c for c in features if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric_features:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric_features}")

    y_reference = df[target].to_numpy(dtype=np.float64)
    if np.any(y_reference < 0):
        raise ValueError(
            f"Target column '{target}' contains negative values. "
            "log1p target transform requires y >= 0."
        )

    X_reference = df[features].to_numpy(dtype=np.float64)
    y_reference_log = np.log1p(y_reference)

    scaler = joblib.load(scaler_path)
    X_reference_scaled = scaler.transform(X_reference).astype(np.float64)

    num_features = len(features)

    with tf.device(device_name):
        model = build_gpflow_gpr(
            X_reference_scaled=X_reference_scaled,
            y_reference_log=y_reference_log,
            num_features=num_features,
        )

        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"Restored GPflow checkpoint from: {latest_ckpt}")

        # Fix all parameters except dmax to training-set medians
        fixed_params = df[features].median().to_dict()

        # Sweep dmax in linspace(0, 1, 11) and remove the 0 entry
        dmax_values = np.linspace(0.0, 0.2, 51, dtype=np.float64)[1:]

        rows = []
        for dmax in dmax_values:
            row = dict(fixed_params)
            row["dmax"] = float(dmax)
            rows.append(row)

        X_eval_df = pd.DataFrame(rows, columns=features)
        X_eval = X_eval_df.to_numpy(dtype=np.float64)
        X_eval_scaled = scaler.transform(X_eval).astype(np.float64)

        mean_f, var_f = model.predict_f(X_eval_scaled)

    y_pred_log = mean_f.numpy().reshape(-1)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0.0, None)

    pred_var_log = var_f.numpy().reshape(-1)
    pred_std_log = np.sqrt(np.clip(pred_var_log, 0.0, None))

    results_df = X_eval_df.copy()
    results_df["pred_eff_area"] = y_pred
    results_df["pred_log_mean"] = y_pred_log
    results_df["pred_log_std"] = pred_std_log

    print("\nFixed parameters used for the sweep:")
    for feature in features:
        if feature != "dmax":
            print(f"  {feature}: {fixed_params[feature]}")

    print("\nPredictions:")
    print(results_df[["dmax", "pred_eff_area", "pred_log_mean", "pred_log_std"]])

    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to: {output_csv.resolve()}")

    plt.figure(figsize=(8, 5))
    plt.plot(results_df["dmax"], results_df["pred_eff_area"], marker="o")
    plt.xlabel("Far-field displacement / dmax")
    plt.ylabel("Predicted eff_area")
    plt.title("GPflow GPR prediction vs far-field displacement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"Saved plot to: {output_png.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
