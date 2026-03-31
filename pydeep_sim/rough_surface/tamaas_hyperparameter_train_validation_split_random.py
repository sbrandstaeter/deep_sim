import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit


def make_predefined_train_validation_split(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    split_column: str = "split",
):
    """
    Combine train and validation dataframes and create a sklearn PredefinedSplit.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    validation_df : pd.DataFrame
        Validation dataframe.
    split_column : str, default="split"
        Name of the column added to the combined dataframe to indicate
        whether a row comes from the training or validation set.

    Returns
    -------
    combined_df : pd.DataFrame
        Concatenation of train_df and validation_df with an added column
        named `split_column` containing either "train" or "validation".
    cv : sklearn.model_selection.PredefinedSplit
        Predefined split object usable as the `cv` argument in GridSearchCV.

    Notes
    -----
    In the PredefinedSplit convention:
    - rows marked with -1 are always used for training
    - rows marked with 0 belong to the validation fold
    """

    train_df = train_df.copy()
    validation_df = validation_df.copy()

    if split_column in train_df.columns:
        raise ValueError(f"Column '{split_column}' already exists in train_df")
    if split_column in validation_df.columns:
        raise ValueError(f"Column '{split_column}' already exists in validation_df")

    train_df[split_column] = "train"
    validation_df[split_column] = "validation"

    combined_df = pd.concat([train_df, validation_df], axis=0, ignore_index=True)

    test_fold = np.where(combined_df[split_column].eq("validation"), 0, -1)
    cv = PredefinedSplit(test_fold=test_fold)

    return combined_df, cv


if __name__ == "__main__":
    from sklearn.model_selection import GridSearchCV
    from sklearn.kernel_ridge import KernelRidge

    # ----------------------------
    # --- Parameters
    # ----------------------------
    n_test = 5625
    seed = 42

    # ----------------------------
    # --- Load data
    # ----------------------------
    train_df = pd.read_csv("tamaas_points_nonperiodic_3_indiv_load_steps.csv")

    # full dataset that will be split into validation + test
    full_eval_df = pd.read_csv("tamaas_points_nonperiodic_3_test_data.csv")

    # ----------------------------
    # --- Split into test + validation
    # ----------------------------
    rng = np.random.default_rng(seed=seed)

    n_total = len(full_eval_df)

    if n_test > n_total:
        raise ValueError(f"n_test ({n_test}) is larger than dataset size ({n_total})")

    test_indices = np.arange(n_test, dtype=int)

    # if you want to choose 5625 final test data at random, uncomment below
    # all_indices = np.arange(n_total)
    # test_indices = rng.choice(all_indices, size=n_test, replace=False)

    mask = np.zeros(n_total, dtype=bool)
    mask[test_indices] = True

    test_df = full_eval_df.iloc[mask].reset_index(drop=True)
    validation_df = full_eval_df.iloc[~mask].reset_index(drop=True)

    print(f"Validation size: {len(validation_df)}")
    print(f"Test size: {len(test_df)}")

    # ----------------------------
    # --- Build predefined split
    # ----------------------------
    combined_df, cv = make_predefined_train_validation_split(train_df, validation_df)

    # you call this before setting everything else up as usual and just continue with the combined_df as if this were the training data
    # cv again can be passed to the tuner via your exisiting structure quite easily
    # important: do you set refit=True (which is the default)?  -> in that case the model is retrained in the end on the entire data so train + validation data, this might be interesting in our case

    # the rest below is just for demonstration purposes:

    # ----------------------------
    # --- Prepare features
    # ----------------------------
    feature_cols = ["dmax"]  # ... all features here

    X = combined_df[feature_cols].to_numpy()
    y = combined_df["target"].to_numpy()

    # ----------------------------
    # --- Hyperparameter search
    # ----------------------------
    param_grid = {
        "alpha": [1e-5, 1e-4, 1e-3],
        "gamma": [1, 5, 10],
        "kernel": ["rbf"],
    }

    grid = GridSearchCV(
        estimator=KernelRidge(),
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        refit=True,
        n_jobs=-1,
    )

    grid.fit(X, y)

    print("Best params:", grid.best_params_)
    print("Best validation score:", grid.best_score_)

    # ----------------------------
    # --- Final model (already refit on train+validation)
    # ----------------------------
    final_model = grid.best_estimator_

    # ----------------------------
    # --- Final test evaluation
    # ----------------------------
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["target"].to_numpy()

    y_pred = final_model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)

    print("Final test MSE:", mse)
