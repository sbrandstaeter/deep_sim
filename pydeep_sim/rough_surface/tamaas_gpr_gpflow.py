#!/usr/bin/env python3
"""
Train and evaluate a GPflow Gaussian Process Regressor on data stored in Parquet format.

Notes:
- Assumes numeric features.
- Uses log1p/expm1 transform on the target, so target values must be >= 0.
- Reports metrics on the original target scale.
- Saves:
    - GPflow model parameters via TensorFlow checkpoint
    - feature scaler via joblib
    - metrics as JSON
- Uses GPU if available and requested.
"""

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from sklearn.preprocessing import StandardScaler

from pydeep_sim.rough_surface.tamaas_gpr_utils import RegressionMetrics


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

    # auto
    if gpus:
        print(f"Using GPU: {gpus[0].name}")
        return "/GPU:0"

    print("No GPU detected by TensorFlow. Falling back to CPU.")
    if not cpus:
        raise RuntimeError("TensorFlow does not report any CPU devices either.")
    return "/CPU:0"


def build_gpflow_gpr(
    X_train_scaled: np.ndarray,
    y_train_log: np.ndarray,
    num_features: int,
) -> gpflow.models.GPR:
    """
    Build a GPflow exact GPR model.

    GPflow expects:
      X: shape [N, D]
      Y: shape [N, P]
    """
    X_tf = X_train_scaled.astype(np.float64)
    y_tf = y_train_log.reshape(-1, 1).astype(np.float64)

    # kernel = gpflow.kernels.Constant(
    #     variance=float(num_features)
    # ) * gpflow.kernels.SquaredExponential(
    #     lengthscales=np.ones(num_features, dtype=np.float64),
    #     variance=1.0,
    # )
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

    # random_state = 20260903
    random_state = 20260331
    rng = np.random.default_rng(random_state)
    drop_na = False
    train_model = True
    device_preference = "gpu"  # "auto", "gpu", or "cpu"

    device_name = configure_tensorflow_device(device_preference)

    experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"

    base_name = f"{experiment_name}_gpr_gpflow_4"

    checkpoint_dir = Path(f"{base_name}_ckpt")
    checkpoint_prefix = str(checkpoint_dir / "ckpt")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = Path(f"{base_name}_scaler.joblib")
    output_path = Path(f"{base_name}_metrics.json")

    train_data_path = Path(f"{experiment_name}.parquet")
    train_data_df_raw = pd.read_parquet(train_data_path)

    test_data_path = Path("tamaas_points_nonperiodic_3_test_data.parquet")
    test_data_df_raw = pd.read_parquet(test_data_path)

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

    num_features = len(features)
    feature_cols = None

    for df_name, df in [("test", test_data_df_raw), ("train", train_data_df_raw)]:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in {df_name} data.")

        if features is None or len(features) == 0:
            feature_cols = [c for c in df.columns if c != target]
        else:
            missing_features = [c for c in features if c not in df.columns]
            if missing_features:
                raise ValueError(
                    f"In {df_name} data, the following feature columns were not found: {missing_features}"
                )
            feature_cols = features

    selected_cols = feature_cols + [target]
    train_data_df = train_data_df_raw[selected_cols].copy()
    test_data_df = test_data_df_raw[selected_cols].copy()

    cleaned_dfs = []
    for df_name, df in [("train", train_data_df), ("test", test_data_df)]:
        if drop_na:
            df = df.dropna()
        else:
            if df.isna().any().any():
                na_counts = df.isna().sum()
                raise ValueError(
                    f"Missing values detected in {df_name} data. "
                    "Either clean the data first or rerun with drop_na=True.\n"
                    f"Missing counts:\n{na_counts[na_counts > 0]}"
                )

        non_numeric_features = [
            c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric_features:
            raise TypeError(
                "GP regression requires numeric features. "
                f"Non-numeric feature columns found in {df_name} data: {non_numeric_features}"
            )

        y_all = df[target].to_numpy()
        if np.any(y_all < 0):
            raise ValueError(
                f"Target column '{target}' contains negative values in {df_name} data. "
                "log1p target transform requires y >= 0."
            )

        cleaned_dfs.append(df)

    train_data_df, test_data_df = cleaned_dfs

    X_train = train_data_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_data_df[target].to_numpy(dtype=np.float64)

    num_train_data = len(y_train)

    test_size = 0.2
    train_size = 1.0 - test_size
    num_test_data = int(num_train_data / train_size * test_size)

    X_test_all = test_data_df[feature_cols].to_numpy(dtype=np.float64)
    y_test_all = test_data_df[target].to_numpy(dtype=np.float64)

    idx_chosen_test_data = rng.choice(
        len(y_test_all), size=num_test_data, replace=False
    )

    # Save the full randomly selected test rows from the original test dataframe
    # (all columns, not only features + target)
    selected_test_rows_path = Path(f"{base_name}_selected_test_rows")
    selected_test_df_full = test_data_df_raw.iloc[idx_chosen_test_data].copy()
    # selected_test_df_full.to_csv(
    #     selected_test_rows_path.with_suffix(".csv"), index=False
    # )
    selected_test_df_full.to_parquet(
        selected_test_rows_path.with_suffix(".parquet"),
        engine="pyarrow",
        compression="zstd",
    )
    print(
        f"Saved selected test data to: {selected_test_rows_path.with_suffix('.parquet')}"
    )

    X_test = X_test_all[idx_chosen_test_data, :]
    y_test = y_test_all[idx_chosen_test_data]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float64)
    X_test_scaled = scaler.transform(X_test).astype(np.float64)

    y_train_log = np.log1p(y_train)

    with tf.device(device_name):
        model = build_gpflow_gpr(
            X_train_scaled=X_train_scaled,
            y_train_log=y_train_log,
            num_features=num_features,
        )

        checkpoint = tf.train.Checkpoint(model=model)

        optimization_time = 0.0
        if train_model:
            start_time = time.time()
            optimizer = gpflow.optimizers.Scipy()
            opt_result = optimizer.minimize(
                model.training_loss,
                variables=model.trainable_variables,
                method="L-BFGS-B",
                options={"maxiter": 1000},
            )
            print("Optimization finished.")
            print(f"Optimizer status: {opt_result.message}")
            optimization_time = time.time() - start_time
            print(f"Optimization took {optimization_time}")

            joblib.dump(scaler, scaler_path)
            print(f"Saved scaler to: {scaler_path.resolve()}")

            saved_ckpt_path = checkpoint.save(checkpoint_prefix)
            print(f"Saved GPflow checkpoint to: {saved_ckpt_path}")
            optimizer_message = str(opt_result.message)
        else:
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_ckpt is None:
                raise FileNotFoundError(
                    f"No checkpoint found in directory: {checkpoint_dir.resolve()}"
                )
            checkpoint.restore(latest_ckpt).expect_partial()
            print(f"Restored GPflow checkpoint from: {latest_ckpt}")
            optimizer_message = None

        mean_f_test, var_f_test = model.predict_f(X_test_scaled)
        mean_f_train, var_f_train = model.predict_f(X_train_scaled)

    y_test_pred_log = mean_f_test.numpy().reshape(-1)
    y_test_pred = np.expm1(y_test_pred_log)
    y_test_pred = np.clip(y_test_pred, 0.0, None)

    y_train_pred_log = mean_f_train.numpy().reshape(-1)
    y_train_pred = np.expm1(y_train_pred_log)
    y_train_pred = np.clip(y_train_pred, 0.0, None)

    regression_metrics_test_data = RegressionMetrics(y_test, y_test_pred)
    regression_metrics_train_data = RegressionMetrics(y_train, y_train_pred)

    metrics = {
        "device_preference": device_preference,
        "device_used": device_name,
        "gpu_visible": len(tf.config.list_physical_devices("GPU")) > 0,
        "n_visible_gpus": len(tf.config.list_physical_devices("GPU")),
        "n_samples_total_train_df": int(len(train_data_df)),
        "n_samples_total_test_df": int(len(test_data_df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_time": optimization_time,
        "target": target,
        "features": feature_cols,
        "test_size": test_size,
        "random_state": random_state,
        "test_data_metrics": regression_metrics_test_data.to_dict(),
        "train_data_metrics": regression_metrics_train_data.to_dict(),
    }

    if optimizer_message is not None:
        metrics["optimizer_message"] = optimizer_message

    print("#" * 20)
    print("Recommended metrics on train data:")
    regression_metrics_train_data.print_recommended_table()
    print("Recommended metrics on test data:")
    regression_metrics_test_data.print_recommended_table()
    print("#" * 20)

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))

    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to: {output_path.resolve()}")

    try:
        print("\nLearned model summary:")
        gpflow.utilities.print_summary(model)
    except Exception as e:
        print("\nCould not print GPflow summary:")
        print(repr(e))
        print("\nLearned kernel:")
        print(model.kernel)
        print("\nLikelihood variance:")
        print(model.likelihood.variance.numpy())


if __name__ == "__main__":
    main()
