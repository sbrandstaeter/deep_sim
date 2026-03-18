#!/usr/bin/env python3
"""
Train and evaluate a Gaussian Process Regressor on data stored in Parquet format.

Usage:
    python train_gpr.py --data data.parquet --target y --features x1 x2 x3

Optional:
    python train_gpr.py --data data.parquet --target y
    # If --features is omitted, all columns except target are used.

Notes:
- Assumes numeric features.
- Uses log1p/expm1 transform on the target, so target values must be >= 0.
- Reports metrics on the original target scale.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Log Error on original-scale predictions.
    Clips predictions at zero because RMSLE requires nonnegative values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0.0, None)

    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def main() -> None:

    random_state = 20260903
    rng = np.random.default_rng(random_state)
    drop_na = False

    experiment_name = "tamaas_points_nonperiodic_3"
    model_path = Path(f"{experiment_name}_gpr_model.joblib")
    output_path = Path(f"{experiment_name}_gpr_model_metrix.json")

    train_data_path = Path(f"{experiment_name}.parquet")
    train_data_df_raw = pd.read_parquet(train_data_path)

    test_data_path = Path(f"{experiment_name}_test_data.parquet")
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

    for df in [test_data_df_raw, train_data_df_raw]:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data {df}.")

        if features is None or len(features) == 0:
            feature_cols = [c for c in df.columns if c != target]
        else:
            missing_features = [c for c in features if c not in df.columns]
            if missing_features:
                raise ValueError(
                    f"In data {df}, the following feature columns were not found: {missing_features}"
                )
            feature_cols = features

    selected_cols = feature_cols + [target]
    train_data_df = train_data_df_raw[selected_cols].copy()
    test_data_df = test_data_df_raw[selected_cols].copy()

    for df in [train_data_df, test_data_df]:
        if drop_na:
            df = df.dropna()
        else:
            if df.isna().any().any():
                na_counts = df.isna().sum()
                raise ValueError(
                    f"Missing values detected in {df}. Either clean the data first or rerun with --drop-na.\n"
                    f"Missing counts:\n{na_counts[na_counts > 0]}"
                )

        non_numeric_features = [
            c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric_features:
            raise TypeError(
                f"GaussianProcessRegressor requires numeric features. "
                f"Non-numeric feature columns found in {df}: {non_numeric_features}"
            )

        y_all = df[target].to_numpy()
        if np.any(y_all < 0):
            raise ValueError(
                f"Target column '{target}' contains negative values in {df}. "
                "log1p target transform requires y >= 0."
            )

    X_train = train_data_df[feature_cols].to_numpy()
    y_train = train_data_df[target].to_numpy()

    num_train_data = len(y_train)

    test_size = 0.2
    train_size = 1 - test_size

    num_test_data = int(num_train_data / train_size * test_size)

    X_test_all = test_data_df[feature_cols].to_numpy()
    y_test_all = test_data_df[target].to_numpy()

    # choose 3 random indices without replacement
    idx_chosen_test_data = rng.choice(
        len(y_test_all), size=num_test_data, replace=False
    )

    X_test = X_test_all[idx_chosen_test_data, :]
    y_test = y_test_all[idx_chosen_test_data]
    # X_train, X_test, y_train, y_test = train_test_split(
    #    X,
    #    y,
    #    test_size=test_size,
    #    random_state=random_state,
    # )

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-3, 1e3)
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e1))

    gpr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=0.0,
                    normalize_y=False,
                    n_restarts_optimizer=3,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=gpr_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True,
    )

    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path.resolve()}")

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, None)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    test_rmsle = rmsle(y_test, y_pred)

    metrics = {
        "n_samples_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": target,
        "features": feature_cols,
        "test_size": test_size,
        "random_state": random_state,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rmsle": test_rmsle,
    }

    print("Evaluation metrics:")
    print(json.dumps(metrics, indent=2))

    output_path.write_text(json.dumps(metrics, indent="2"), encoding="utf-8")
    print(f"\nSaved metrics to: {output_path.resolve()}")

    trained_gpr = model.regressor_.named_steps["gpr"]
    print("\nLearned kernel:")
    print(trained_gpr.kernel_)


if __name__ == "__main__":
    main()
