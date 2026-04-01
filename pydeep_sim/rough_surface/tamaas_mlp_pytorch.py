#!/usr/bin/env python3
"""
Train and evaluate a PyTorch feedforward neural network regressor on data stored in
Parquet format, using a structured validation split based on whole ids.

Properties:
- structured validation split by ids
- 50 hidden layers
- 100 neurons per hidden layer
- configurable activation
- biased linear layers
- SGD optimizer
- adaptive LR via ReduceLROnPlateau
- log1p/expm1 target transform

Notes:
- Assumes numeric features.
- Uses log1p/expm1 transform on the target, so target values must be >= 0.
- Reports metrics on the original target scale.
- Uses CUDA automatically if available.
"""

import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pydeep_sim.rough_surface.tamaas_hyperparameter_train_validation_split import (
    cv_train_validation_splits,
)
from pydeep_sim.rough_surface.tamaas_ml_model_utils import RegressionMetrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeedForwardRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        bias: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if activation == "sigmoid":
            activation_layer = nn.Sigmoid
        elif activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "tanh":
            activation_layer = nn.Tanh
        elif activation == "silu":
            activation_layer = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(activation_layer())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1, bias=bias))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@torch.no_grad()
def predict_in_batches(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []

    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        yb_pred = model(xb).squeeze(-1)
        preds.append(yb_pred.detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)

            batch_size = xb.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def main() -> None:
    random_state = 20260903
    set_seed(random_state)

    rng = np.random.default_rng(random_state)
    drop_na = False

    experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
    base_name = f"{experiment_name}_mlp_torch_structured_val_1"

    model_path = Path(f"{base_name}_model.pt")
    scaler_path = Path(f"{base_name}_x_scaler.joblib")
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

    # Structured validation options
    n_structured_val_sets = 5
    val_fold_idx = 0  # choose 0..n_structured_val_sets-1

    # Model options
    hidden_dims = [64, 32]
    activation = "relu"
    bias = True

    # Optimization options
    batch_size_gpu = 8192
    batch_size_cpu = 1024
    learning_rate = 0.1
    weight_decay = 0.001
    max_epochs = 5000
    early_stopping_patience = 10

    feature_cols = None

    for df_name, df in [("test", test_data_df_raw), ("train", train_data_df_raw)]:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in {df_name} data.")

        if features is None or len(features) == 0:
            feature_cols = [c for c in df.columns if c not in [target, "ids"]]
        else:
            missing_features = [c for c in features if c not in df.columns]
            if missing_features:
                raise ValueError(
                    f"In {df_name} data, the following feature columns were not found: "
                    f"{missing_features}"
                )
            feature_cols = features

    if "ids" not in train_data_df_raw.columns:
        raise ValueError("Training dataframe must contain an 'ids' column")

    selected_cols_train = ["ids"] + feature_cols + [target]
    selected_cols_test = feature_cols + [target]

    train_data_df = train_data_df_raw[selected_cols_train].copy()
    test_data_df = test_data_df_raw[selected_cols_test].copy()

    cleaned_train_df = train_data_df.copy()
    cleaned_test_df = test_data_df.copy()

    if drop_na:
        cleaned_train_df = cleaned_train_df.dropna()
        cleaned_test_df = cleaned_test_df.dropna()
    else:
        if cleaned_train_df.isna().any().any():
            na_counts = cleaned_train_df.isna().sum()
            raise ValueError(
                "Missing values detected in train data. "
                "Either clean the data first or rerun with drop_na=True.\n"
                f"Missing counts:\n{na_counts[na_counts > 0]}"
            )
        if cleaned_test_df.isna().any().any():
            na_counts = cleaned_test_df.isna().sum()
            raise ValueError(
                "Missing values detected in test data. "
                "Either clean the data first or rerun with drop_na=True.\n"
                f"Missing counts:\n{na_counts[na_counts > 0]}"
            )

    non_numeric_train_features = [
        c
        for c in feature_cols
        if not pd.api.types.is_numeric_dtype(cleaned_train_df[c])
    ]
    if non_numeric_train_features:
        raise TypeError(
            "Neural-network regression requires numeric features. "
            f"Non-numeric feature columns found in train data: {non_numeric_train_features}"
        )

    non_numeric_test_features = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(cleaned_test_df[c])
    ]
    if non_numeric_test_features:
        raise TypeError(
            "Neural-network regression requires numeric features. "
            f"Non-numeric feature columns found in test data: {non_numeric_test_features}"
        )

    y_train_all_check = cleaned_train_df[target].to_numpy()
    if np.any(y_train_all_check < 0):
        raise ValueError(
            f"Target column '{target}' contains negative values in train data. "
            "log1p target transform requires y >= 0."
        )

    y_test_all_check = cleaned_test_df[target].to_numpy()
    if np.any(y_test_all_check < 0):
        raise ValueError(
            f"Target column '{target}' contains negative values in test data. "
            "log1p target transform requires y >= 0."
        )

    train_data_df = cleaned_train_df
    test_data_df = cleaned_test_df

    # ------------------------------------------------------------------
    # Structured validation split from train data using whole ids
    # ------------------------------------------------------------------
    cv_splits = cv_train_validation_splits(
        train_data_df,
        n_sets=n_structured_val_sets,
        seed=random_state,
    )

    if not (0 <= val_fold_idx < len(cv_splits)):
        raise ValueError(
            f"val_fold_idx must be in [0, {len(cv_splits) - 1}], got {val_fold_idx}"
        )

    train_idx, val_idx = cv_splits[val_fold_idx]

    train_subset_df = train_data_df.iloc[train_idx].copy()
    val_subset_df = train_data_df.iloc[val_idx].copy()

    print(
        f"Using structured validation fold {val_fold_idx + 1}/{n_structured_val_sets}"
    )
    print(f"Structured train rows: {len(train_subset_df)}")
    print(f"Structured val rows: {len(val_subset_df)}")
    print(f"Structured train ids: {train_subset_df['ids'].nunique()}")
    print(f"Structured val ids: {val_subset_df['ids'].nunique()}")

    # ------------------------------------------------------------------
    # Final train / val / test arrays
    # ------------------------------------------------------------------
    X_tr_raw = train_subset_df[feature_cols].to_numpy(dtype=np.float32)
    y_tr_raw = train_subset_df[target].to_numpy(dtype=np.float32)

    X_val_raw = val_subset_df[feature_cols].to_numpy(dtype=np.float32)
    y_val_raw = val_subset_df[target].to_numpy(dtype=np.float32)

    X_train_full_raw = train_data_df[feature_cols].to_numpy(dtype=np.float32)
    y_train_full = train_data_df[target].to_numpy(dtype=np.float32)

    num_train_data = len(y_train_full)

    test_size = 0.2
    train_size = 1.0 - test_size
    num_test_data = int(num_train_data / train_size * test_size)

    X_test_all_raw = test_data_df[feature_cols].to_numpy(dtype=np.float32)
    y_test_all = test_data_df[target].to_numpy(dtype=np.float32)

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
        f"Saved selected test data to: "
        f"{selected_test_rows_path.with_suffix('.parquet')}"
    )

    X_test_raw = X_test_all_raw[idx_chosen_test_data, :]
    y_test = y_test_all[idx_chosen_test_data]

    # Fit scaler only on structured training subset
    x_scaler = StandardScaler()
    X_tr = x_scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_val = x_scaler.transform(X_val_raw).astype(np.float32)
    X_train_full_scaled = x_scaler.transform(X_train_full_raw).astype(np.float32)
    X_test = x_scaler.transform(X_test_raw).astype(np.float32)

    joblib.dump(x_scaler, scaler_path)
    print(f"Saved feature scaler to: {scaler_path.resolve()}")

    # log1p target transform
    y_tr = np.log1p(y_tr_raw).astype(np.float32)
    y_val = np.log1p(y_val_raw).astype(np.float32)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    if use_cuda:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    batch_size = batch_size_gpu if use_cuda else batch_size_cpu
    num_workers = 4 if use_cuda else 0
    pin_memory = use_cuda

    train_ds = TensorDataset(
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )
    test_pred_ds = TensorDataset(torch.from_numpy(X_test))
    train_full_pred_ds = TensorDataset(torch.from_numpy(X_train_full_scaled))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_pred_loader = DataLoader(
        test_pred_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    train_full_pred_loader = DataLoader(
        train_full_pred_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    model = FeedForwardRegressor(
        input_dim=X_tr.shape[1],
        hidden_dims=hidden_dims,
        bias=bias,
        activation=activation,
    ).to(device)

    print("\nModel structure:")
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.0,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6,
    )

    best_val_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    best_state = None

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            batch_size_actual = xb.shape[0]
            running_loss += float(loss.item()) * batch_size_actual
            running_count += batch_size_actual

        train_loss = running_loss / max(running_count, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:4d} | "
            f"train_loss={train_loss:.8f} | "
            f"val_loss={val_loss:.8f} | "
            f"lr={current_lr:.6g}"
        )

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "epoch": epoch,
                "val_loss": val_loss,
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {epoch} epochs. "
                    f"Best epoch: {best_epoch}"
                )
                break

    optimization_time = time.time() - start_time
    print("Training finished.")
    print(f"Training took {optimization_time:.3f} seconds")

    if best_state is None:
        raise RuntimeError(
            "Training finished without producing a valid best model state."
        )

    model.load_state_dict(best_state["model_state_dict"])

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": X_tr.shape[1],
        "hidden_dims": hidden_dims,
        "bias": bias,
        "activation": activation,
        "feature_cols": feature_cols,
        "target": target,
        "random_state": random_state,
        "n_structured_val_sets": n_structured_val_sets,
        "val_fold_idx": val_fold_idx,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, model_path)
    print(f"Saved model to: {model_path.resolve()}")

    # Predictions on original scale
    y_test_pred_log = predict_in_batches(model, test_pred_loader, device)
    y_test_pred = np.expm1(y_test_pred_log)
    y_test_pred = np.clip(y_test_pred, 0.0, None)

    y_train_full_pred_log = predict_in_batches(model, train_full_pred_loader, device)
    y_train_full_pred = np.expm1(y_train_full_pred_log)
    y_train_full_pred = np.clip(y_train_full_pred, 0.0, None)

    regression_metrics_test_data = RegressionMetrics(y_test, y_test_pred)
    regression_metrics_train_data = RegressionMetrics(y_train_full, y_train_full_pred)

    metrics = {
        "n_samples_total_train_df": int(len(train_data_df)),
        "n_samples_total_test_df": int(len(test_data_df)),
        "n_train_full": int(len(X_train_full_raw)),
        "n_train_structured": int(len(X_tr)),
        "n_val_structured": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_train_ids_total": int(train_data_df["ids"].nunique()),
        "n_train_ids_structured": int(train_subset_df["ids"].nunique()),
        "n_val_ids_structured": int(val_subset_df["ids"].nunique()),
        "train_time": optimization_time,
        "target": target,
        "features": feature_cols,
        "test_size": test_size,
        "random_state": random_state,
        "validation_split_type": "structured_ids",
        "n_structured_val_sets": n_structured_val_sets,
        "val_fold_idx": val_fold_idx,
        "model_type": "PyTorchFeedForwardRegressor",
        "hidden_dims": hidden_dims,
        "bias": bias,
        "activation": activation,
        "optimizer": "SGD",
        "weight_decay": weight_decay,
        "adaptive_lr_scheduler": "ReduceLROnPlateau",
        "initial_lr": learning_rate,
        "max_epochs": max_epochs,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "device": str(device),
        "cuda_available": bool(use_cuda),
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


if __name__ == "__main__":
    main()
