#!/usr/bin/env python3
"""
Train and evaluate an MLP Regressor on data stored in Parquet format.

Notes:
- Assumes numeric features.
- Uses log1p/expm1 transform on the target, so target values must be >= 0.
- Reports metrics on the original target scale.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pydeep_sim.rough_surface.tamaas_ml_model_utils import RegressionMetrics


def main() -> None:
    random_state = 20260903
    rng = np.random.default_rng(random_state)
    drop_na = False

    experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"

    base_name = f"{experiment_name}_mlp_scikit_1"

    model_path = Path(f"{base_name}_model.joblib")
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
                "MLP regression requires numeric features. "
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

    X_train = train_data_df[feature_cols].to_numpy()
    y_train = train_data_df[target].to_numpy()

    num_train_data = len(y_train)

    test_size = 0.2
    train_size = 1.0 - test_size
    num_test_data = int(num_train_data / train_size * test_size)

    X_test_all = test_data_df[feature_cols].to_numpy()
    y_test_all = test_data_df[target].to_numpy()

    idx_chosen_test_data = rng.choice(
        len(y_test_all), size=num_test_data, replace=False
    )

    selected_test_rows_path = Path(f"{base_name}_selected_test_rows")
    selected_test_df_full = test_data_df_raw.iloc[idx_chosen_test_data].copy()
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

    hidden_layers = (100,) * 50

    mlp_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation="logistic",
                    solver="sgd",
                    alpha=0.001,
                    learning_rate="adaptive",
                    learning_rate_init=0.01,
                    max_iter=200,
                    shuffle=True,
                    random_state=random_state,
                    verbose=True,
                    early_stopping=True,
                    n_iter_no_change=10,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=mlp_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=True,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    print("Training finished.")
    optimization_time = time.time() - start_time
    print(f"Training took {optimization_time}")

    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path.resolve()}")

    y_test_pred = model.predict(X_test)
    y_test_pred = np.clip(y_test_pred, 0.0, None)

    y_train_pred = model.predict(X_train)
    y_train_pred = np.clip(y_train_pred, 0.0, None)

    trained_mlp = model.regressor_.named_steps["mlp"]

    regression_metrics_test_data = RegressionMetrics(y_test, y_test_pred)
    regression_metrics_train_data = RegressionMetrics(y_train, y_train_pred)

    metrics = {
        "n_samples_total_train_df": int(len(train_data_df)),
        "n_samples_total_test_df": int(len(test_data_df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_time": optimization_time,
        "target": target,
        "features": feature_cols,
        "test_size": test_size,
        "random_state": random_state,
        "model_type": "MLPRegressor",
        "hidden_layer_sizes": hidden_layers,
        "activation": "logistic",
        "solver": "sgd",
        "alpha": 0.001,
        "learning_rate": "adaptive",
        "learning_rate_init": 0.01,
        "max_iter": 200,
        "n_iter_": int(trained_mlp.n_iter_),
        "loss": float(trained_mlp.loss_),
        "test_data_metrics": regression_metrics_test_data.to_dict(),
        "train_data_metrics": regression_metrics_train_data.to_dict(),
    }

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

    print("\nTrained MLP summary:")
    print(f"Iterations: {trained_mlp.n_iter_}")
    print(f"Final loss: {trained_mlp.loss_}")


if __name__ == "__main__":
    main()
