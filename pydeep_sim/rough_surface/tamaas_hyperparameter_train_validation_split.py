import numpy as np
import pandas as pd


def cv_train_validation_splits(df: pd.DataFrame, n_sets: int = 5, seed: int = 42):
    """
    Compute train and validation splits of data.

    df contains the raw data prior to removing all non-used columns (it needs the ids column).
    """
    # --- Validate ids column exists ---
    if "ids" not in df.columns:
        raise ValueError("Dataframe must contain a column named 'ids'")

    # --- Count rows per id ---
    id_counts = df["ids"].value_counts().sort_index()

    if id_counts.empty:
        raise ValueError("The dataframe is empty")

    # --- Check all ids have the same number of rows ---
    if id_counts.nunique() != 1:
        raise ValueError(
            "Not all ids have the same number of rows.\n"
            f"Found counts: {id_counts.unique()}"
        )

    rows_per_id = id_counts.iloc[0]
    unique_ids = id_counts.index.to_numpy()
    n_ids = len(unique_ids)

    print(f"Found {n_ids} unique ids")
    print(f"Each id appears {rows_per_id} times")

    if n_ids % n_sets != 0:
        raise ValueError(f"Number of unique ids ({n_ids}) is not divisible by {n_sets}")

    ids_per_set = n_ids // n_sets

    # --- Randomly split unique ids into 5 equal sets ---
    rng = np.random.default_rng(seed=seed)  # optional seed for reproducibility

    shuffled_ids = rng.permutation(unique_ids)
    id_sets = np.split(shuffled_ids, n_sets)

    # --- Build custom CV splits as row indices ---
    cv_splits = []
    for test_ids in id_sets:
        test_mask = df["ids"].isin(test_ids).to_numpy()
        test_idx = np.flatnonzero(test_mask)
        train_idx = np.flatnonzero(~test_mask)
        cv_splits.append((train_idx, test_idx))

    return cv_splits


if __name__ == "__main__":
    # --- Load dataframe from disk ---"
    file_path = "tamaas_points_nonperiodic_3_indiv_load_steps.csv"

    n_sets = 5  # this is equal to k for k-fold cross validation

    seed = 42

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

    cv_splits = cv_train_validation_splits(df, n_sets=n_sets, seed=seed)

    # according to https://github.com/imcs-compsim/deep_sim/blob/master/pydeep_sim/machine_learning/model/regression_model.py#L171
    # the following should now easily work:
    tuner_options = {}
    tuner_options["cv"] = cv_splits

    # this can be easily set at the place where self.model_block["parameter_tuning"]["options"] is defined.

    ## see the example below
    # # --- Define features and target ---
    # feature_cols = [c for c in df.columns if c not in ["ids", "target"]]
    # X = df[feature_cols].to_numpy()
    # y = df["target"].to_numpy()
    #
    # # --- Example model + grid search ---
    # param_grid = {
    #     "alpha": [1e-5, 1e-4, 1e-3],
    #     "gamma": [1, 5, 10],
    #     "kernel": ["rbf"],
    # }
    #
    # grid = GridSearchCV(
    #     estimator=KernelRidge(),
    #     param_grid=param_grid,
    #     cv=cv_splits,   # <- your manual folds go here
    #     scoring="neg_mean_squared_error",
    #     n_jobs=-1,
    # )
    #
    # grid.fit(X, y)
    #
    # print("Best params:", grid.best_params_)
    # print("Best CV score:", grid.best_score_)
