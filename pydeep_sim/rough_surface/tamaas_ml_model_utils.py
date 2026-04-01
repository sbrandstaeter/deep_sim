from dataclasses import dataclass, field

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error as sk_max_error,
)


def rmsle(y_true, y_pred):
    """
    Root Mean Squared Log Error on original-scale predictions.
    Clips predictions at zero because RMSLE requires nonnegative values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("RMSLE cannot be computed with negative values.")

    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


@dataclass
class RegressionMetrics:
    y_true: np.ndarray
    y_pred: np.ndarray

    # reference statistics
    mean_ground_truth: float = field(init=False)
    mean_abs_ground_truth: float = field(init=False)
    max_abs_ground_truth: float = field(init=False)
    range_ground_truth: float = field(init=False)
    var_ground_truth: float = field(init=False)
    std_ground_truth: float = field(init=False)

    # standard metrics
    mse: float = field(init=False)
    rmse: float = field(init=False)
    mae: float = field(init=False)
    max_error: float = field(init=False)
    r2: float = field(init=False)
    rmsle: float | None = field(init=False)

    # normalization by mean
    nmse_mean: float | None = field(init=False)
    nrmse_mean: float | None = field(init=False)
    nmae_mean: float | None = field(init=False)
    nrmsle_mean: float | None = field(init=False)

    # normalization by mean abs
    nmse_mean_abs: float | None = field(init=False)
    nrmse_mean_abs: float | None = field(init=False)
    nmae_mean_abs: float | None = field(init=False)
    nrmsle_mean_abs: float | None = field(init=False)

    # normalization by range
    nmse_range: float | None = field(init=False)
    nrmse_range: float | None = field(init=False)
    nmae_range: float | None = field(init=False)
    nrmsle_range: float | None = field(init=False)

    # normalization by variance
    nmse_var: float | None = field(init=False)
    nrmse_var: float | None = field(init=False)
    nmae_var: float | None = field(init=False)
    nrmsle_var: float | None = field(init=False)

    # normalization by std
    nrmse_std: float | None = field(init=False)

    # normalization by max abs
    nmse_max_abs: float | None = field(init=False)
    nrmse_max_abs: float | None = field(init=False)
    nmae_max_abs: float | None = field(init=False)
    nrmsle_max_abs: float | None = field(init=False)
    nmax_error_max_abs: float | None = field(init=False)

    def __post_init__(self):
        self.y_true = np.asarray(self.y_true, dtype=float)
        self.y_pred = np.asarray(self.y_pred, dtype=float)

        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        if self.y_true.size == 0:
            raise ValueError("y_true and y_pred must not be empty.")

        # reference statistics
        self.mean_ground_truth = float(np.mean(self.y_true))
        self.mean_abs_ground_truth = float(np.mean(np.abs(self.y_true)))
        self.max_abs_ground_truth = float(np.max(np.abs(self.y_true)))
        self.range_ground_truth = float(np.max(self.y_true) - np.min(self.y_true))
        self.var_ground_truth = float(np.var(self.y_true))
        self.std_ground_truth = float(np.std(self.y_true))

        # standard metrics
        self.mse = float(mean_squared_error(self.y_true, self.y_pred))
        self.rmse = float(np.sqrt(self.mse))
        self.mae = float(mean_absolute_error(self.y_true, self.y_pred))
        self.max_error = float(sk_max_error(self.y_true, self.y_pred))
        self.r2 = float(r2_score(self.y_true, self.y_pred))

        try:
            self.rmsle = rmsle(self.y_true, self.y_pred)
        except ValueError:
            self.rmsle = None

        # normalization by mean
        self.nmse_mean = self._safe_div(self.mse, self.mean_ground_truth)
        self.nrmse_mean = self._safe_div(self.rmse, self.mean_ground_truth)
        self.nmae_mean = self._safe_div(self.mae, self.mean_ground_truth)
        self.nrmsle_mean = self._safe_div(self.rmsle, self.mean_ground_truth)

        # normalization by mean abs
        self.nmse_mean_abs = self._safe_div(self.mse, self.mean_abs_ground_truth)
        self.nrmse_mean_abs = self._safe_div(self.rmse, self.mean_abs_ground_truth)
        self.nmae_mean_abs = self._safe_div(self.mae, self.mean_abs_ground_truth)
        self.nrmsle_mean_abs = self._safe_div(self.rmsle, self.mean_abs_ground_truth)

        # normalization by range
        self.nmse_range = self._safe_div(self.mse, self.range_ground_truth)
        self.nrmse_range = self._safe_div(self.rmse, self.range_ground_truth)
        self.nmae_range = self._safe_div(self.mae, self.range_ground_truth)
        self.nrmsle_range = self._safe_div(self.rmsle, self.range_ground_truth)

        # normalization by variance
        self.nmse_var = self._safe_div(self.mse, self.var_ground_truth)
        self.nrmse_var = self._safe_div(self.rmse, self.var_ground_truth)
        self.nmae_var = self._safe_div(self.mae, self.var_ground_truth)
        self.nrmsle_var = self._safe_div(self.rmsle, self.var_ground_truth)

        # normalization by std
        self.nrmse_std = self._safe_div(self.rmse, self.std_ground_truth)

        # normalization by max abs
        self.nmse_max_abs = self._safe_div(self.mse, self.max_abs_ground_truth)
        self.nrmse_max_abs = self._safe_div(self.rmse, self.max_abs_ground_truth)
        self.nmae_max_abs = self._safe_div(self.mae, self.max_abs_ground_truth)
        self.nrmsle_max_abs = self._safe_div(self.rmsle, self.max_abs_ground_truth)
        self.nmax_error_max_abs = self._safe_div(
            self.max_error, self.max_abs_ground_truth
        )

    @staticmethod
    def _safe_div(numerator, denominator):
        if numerator is None or denominator == 0:
            return None
        return float(numerator / denominator)

    def get_recommended_metrics(self):
        return {
            "r2": self.r2,
            "nmse_var": self.nmse_var,
            "nrmse_std": self.nrmse_std,
            "nmae_mean_abs": self.nmae_mean_abs,
            "nmax_error_max_abs": self.nmax_error_max_abs,
            "rmsle": self.rmsle,
        }

    def to_dict(self):
        return {
            "mean_ground_truth": self.mean_ground_truth,
            "mean_abs_ground_truth": self.mean_abs_ground_truth,
            "max_abs_ground_truth": self.max_abs_ground_truth,
            "range_ground_truth": self.range_ground_truth,
            "var_ground_truth": self.var_ground_truth,
            "std_ground_truth": self.std_ground_truth,
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_error": self.max_error,
            "r2": self.r2,
            "rmsle": self.rmsle,
            "nmse_mean": self.nmse_mean,
            "nrmse_mean": self.nrmse_mean,
            "nmae_mean": self.nmae_mean,
            "nrmsle_mean": self.nrmsle_mean,
            "nmse_mean_abs": self.nmse_mean_abs,
            "nrmse_mean_abs": self.nrmse_mean_abs,
            "nmae_mean_abs": self.nmae_mean_abs,
            "nrmsle_mean_abs": self.nrmsle_mean_abs,
            "nmse_range": self.nmse_range,
            "nrmse_range": self.nrmse_range,
            "nmae_range": self.nmae_range,
            "nrmsle_range": self.nrmsle_range,
            "nmse_var": self.nmse_var,
            "nrmse_var": self.nrmse_var,
            "nmae_var": self.nmae_var,
            "nrmsle_var": self.nrmsle_var,
            "nrmse_std": self.nrmse_std,
            "nmse_max_abs": self.nmse_max_abs,
            "nrmse_max_abs": self.nrmse_max_abs,
            "nmae_max_abs": self.nmae_max_abs,
            "nrmsle_max_abs": self.nrmsle_max_abs,
            "nmax_error_max_abs": self.nmax_error_max_abs,
            "recommended_metrics": self.get_recommended_metrics(),
        }

    @staticmethod
    def _format_value(value, precision=6):
        if value is None:
            return "n/a"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{precision}g}"
        return str(value)

    def print_table(self, precision=6):
        rows = [(k, v) for k, v in self.to_dict().items() if k != "recommended_metrics"]

        name_width = max(len(name) for name, _ in rows)
        value_width = max(
            len(self._format_value(value, precision)) for _, value in rows
        )

        line = f"+-{'-' * name_width}-+-{'-' * value_width}-+"
        print(line)
        print(f"| {'metric'.ljust(name_width)} | {'value'.ljust(value_width)} |")
        print(line)

        for name, value in rows:
            formatted = self._format_value(value, precision)
            print(f"| {name.ljust(name_width)} | {formatted.ljust(value_width)} |")

        print(line)

    def print_recommended_table(self, precision=6):
        rows = list(self.get_recommended_metrics().items())

        name_width = max(len(name) for name, _ in rows)
        value_width = max(
            len(self._format_value(value, precision)) for _, value in rows
        )

        line = f"+-{'-' * name_width}-+-{'-' * value_width}-+"
        print(line)
        print(
            f"| {'recommended_metric'.ljust(name_width)} | {'value'.ljust(value_width)} |"
        )
        print(line)

        for name, value in rows:
            formatted = self._format_value(value, precision)
            print(f"| {name.ljust(name_width)} | {formatted.ljust(value_width)} |")

        print(line)

    def regression_metrics_to_latex(self, caption: str = "", label: str = "") -> str:
        # Adjust keys depending on your actual RegressionMetrics implementation

        rows = [
            ("nMAE", self.nmae_mean_abs),
            ("nMSE", self.nmse_var),
            ("nMaxE", self.nmax_error_max_abs),
            ("R$^2$", self.r2),
        ]

        # Filter out None values (in case some metrics are missing)
        rows = [(name, val) for name, val in rows if val is not None]

        table = []
        table.append("\\begin{table}[h!]")
        table.append("\\centering")
        table.append("\\begin{tabular}{lr}")
        table.append("\\toprule")
        table.append("Metric & Value \\\\")
        table.append("\\midrule")

        for name, val in rows:
            table.append(f"{name} & {val:.6g} \\\\")

        table.append("\\bottomrule")
        table.append("\\end{tabular}")

        if caption:
            table.append(f"\\caption{{{caption}}}")
        if label:
            table.append(f"\\label{{{label}}}")

        table.append("\\end{table}")

        return "\n".join(table)
