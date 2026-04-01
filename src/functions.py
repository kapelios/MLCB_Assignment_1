import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from a given path.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    return pd.read_csv(path)

def split_data(
    df: pd.DataFrame,
    test_size: float,
    stratify_by: str,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into training and validation sets.

    Args:
        df (pd.DataFrame): The dataframe to split.
        test_size (float): The proportion of the dataset to include in the validation split.
        stratify_by (str): The column to use for stratification.
        random_state (int): The seed used by the random number generator.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and validation dataframes.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_by],
        random_state=random_state
    )
    return train_df, val_df

def create_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create age bins for stratification.

    Args:
        df (pd.DataFrame): The dataframe with an 'age' column.

    Returns:
        pd.DataFrame: The dataframe with a new 'age_bin' column.
    """
    df = df.copy()
    bins = [0, 30, 40, 50, 60, 70, 80, float('inf')]
    labels = ['0-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df

def get_feature_names(df: pd.DataFrame) -> Tuple[list, list]:
    """Get lists of CpG and categorical feature names.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        Tuple[list, list]: A tuple containing lists of CpG and categorical feature names.
    """
    cpg_features = [col for col in df.columns if col.startswith('cg')]
    categorical_features = [col for col in ['sex', 'ethnicity'] if col in df.columns]
    return cpg_features, categorical_features


def create_preprocessor(cpg_features: list, categorical_features: list) -> ColumnTransformer:
    """Create a preprocessor for the data.

    Args:
        cpg_features (list): List of CpG feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        ColumnTransformer: The preprocessor.
    """
    cpg_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cpg', cpg_pipeline, cpg_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def train_and_evaluate_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    preprocessor: ColumnTransformer,
    n_bootstraps: int = 1000,
    seed: int = 42,
    model=None,
    return_raw: bool = False
) -> dict:
    """Train a model and evaluate it using bootstrap resampling.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        val_df (pd.DataFrame): The validation dataframe.
        feature_cols (list): The list of feature columns to use.
        target_col (str): The name of the target column.
        preprocessor (ColumnTransformer): The preprocessor to use.
        n_bootstraps (int, optional): The number of bootstrap samples. Defaults to 1000.
        seed (int, optional): The random seed. Defaults to 42.

    Returns:
        dict: A dictionary with the performance metrics.
    """
    # Define the model
    if model is None:
        model = LinearRegression()

    # Create the full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    # Separate features and target
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = pipeline.predict(X_val)

    # Bootstrap resampling for performance metrics
    np.random.seed(seed)
    metrics = {'rmse': [], 'mae': [], 'r2': [], 'pearsonr': []}
    for _ in range(n_bootstraps):
        indices = np.random.choice(range(len(y_val)), len(y_val), replace=True)
        y_val_sample = y_val.iloc[indices]
        y_pred_sample = y_pred[indices]

        metrics['rmse'].append(np.sqrt(mean_squared_error(y_val_sample, y_pred_sample)))
        metrics['mae'].append(mean_absolute_error(y_val_sample, y_pred_sample))
        metrics['r2'].append(r2_score(y_val_sample, y_pred_sample))
        metrics['pearsonr'].append(pearsonr(y_val_sample, y_pred_sample)[0])

    # Calculate 95% confidence intervals
    results = {}
    for metric_name, values in metrics.items():
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        mean = np.mean(values)
        results[metric_name] = f"{mean:.3f} (95% CI: {lower:.3f}-{upper:.3f})"

    if return_raw:
        return results, {k: np.array(v) for k, v in metrics.items()}
    return results


def bootstrap_evaluate(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstraps: int = 1000,
    seed: int = 42,
    return_raw: bool = False,
) -> dict:
    """Bootstrap evaluation of a pre-fitted pipeline on a held-out set.

    Unlike train_and_evaluate_model, this function expects an already-fitted
    pipeline and does not perform any training. Use this for final evaluation
    on the locked evaluation set.

    Args:
        pipeline: Already-fitted sklearn Pipeline or estimator.
        X: Feature DataFrame for evaluation.
        y: Target Series for evaluation.
        n_bootstraps: Number of bootstrap resamples.
        seed: Random seed.
        return_raw: If True, also return dict of raw bootstrap arrays.

    Returns:
        Dict mapping metric name -> formatted string "mean (95% CI: lo-hi)".
        If return_raw=True, returns (results_dict, raw_arrays_dict).
    """
    y_pred = pipeline.predict(X)

    np.random.seed(seed)
    metrics = {'rmse': [], 'mae': [], 'r2': [], 'pearsonr': []}
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y), len(y), replace=True)
        y_s = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
        y_p = y_pred[idx]
        metrics['rmse'].append(np.sqrt(mean_squared_error(y_s, y_p)))
        metrics['mae'].append(mean_absolute_error(y_s, y_p))
        metrics['r2'].append(r2_score(y_s, y_p))
        metrics['pearsonr'].append(pearsonr(y_s, y_p)[0])

    results = {}
    for metric_name, values in metrics.items():
        arr = np.array(values)
        lower = np.percentile(arr, 2.5)
        upper = np.percentile(arr, 97.5)
        mean  = np.mean(arr)
        results[metric_name] = f"{mean:.3f} (95% CI: {lower:.3f}-{upper:.3f})"

    if return_raw:
        return results, {k: np.array(v) for k, v in metrics.items()}
    return results


def stability_selection(
    train_df: pd.DataFrame,
    cpg_features: list,
    target_col: str = 'age',
    n_subsamples: int = 50,
    subsample_frac: float = 0.8,
    top_k: int = 200,
    threshold: float = 0.5,
    seed: int = 42
) -> Tuple[list, dict]:
    """Stability selection via Spearman correlation ranking.

    For each subsample, ranks CpG features by |Spearman r| with target and
    keeps the top_k. A feature is considered stable if it appears in more than
    threshold * n_subsamples resamples.

    Args:
        train_df: Training dataframe.
        cpg_features: List of CpG feature column names.
        target_col: Name of the target column.
        n_subsamples: Number of subsamples to draw.
        subsample_frac: Fraction of training data per subsample (no replacement).
        top_k: Number of top features to retain per subsample.
        threshold: Minimum selection frequency (fraction) to be deemed stable.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (stable_features list, selection_frequency dict mapping
        feature name -> selection count across subsamples).
    """
    # Mean-impute CpG features on the full training set (MCAR assumption,
    # consistent with the preprocessing pipeline in create_preprocessor).
    imputer = SimpleImputer(strategy='mean')
    X_all = imputer.fit_transform(train_df[cpg_features])

    rng = np.random.default_rng(seed)
    n_samples = len(train_df)
    subsample_size = int(np.floor(subsample_frac * n_samples))

    selection_counts = {f: 0 for f in cpg_features}
    y = train_df[target_col].values

    for _ in range(n_subsamples):
        idx = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X_all[idx]
        y_sub = y[idx]

        # Compute |Spearman r| for each feature (no NaN after imputation)
        corrs = np.array([
            abs(spearmanr(X_sub[:, j], y_sub).statistic)
            for j in range(X_sub.shape[1])
        ])

        # Keep indices of top_k features by |r|
        top_idx = np.argpartition(corrs, -top_k)[-top_k:]
        for i in top_idx:
            selection_counts[cpg_features[i]] += 1

    min_count = threshold * n_subsamples
    stable_features = [f for f, cnt in selection_counts.items() if cnt > min_count]
    return stable_features, selection_counts


def parse_results_table(task_results: dict) -> pd.DataFrame:
    """Parse bootstrap result strings from train_and_evaluate_model into a summary DataFrame.

    Args:
        task_results: Dict mapping model name -> metrics dict (values are
            formatted strings like "4.911 (95% CI: 4.355-5.458)").

    Returns:
        DataFrame indexed by model name with columns:
        RMSE Mean, RMSE 95% CI, MAE Mean, MAE 95% CI, R², R² 95% CI,
        Pearson r Mean, Pearson r 95% CI.
    """
    rows = []
    for model_name, metrics in task_results.items():
        row = {"Model": model_name}
        for metric, value in metrics.items():
            mean_val = float(re.match(r"([\d.\-]+)", value).group(1))
            ci_match = re.search(r"\((.*?)\)", value)
            ci_str = ci_match.group(1).replace("95% CI: ", "") if ci_match else ""
            if metric == "rmse":
                row["RMSE Mean"] = mean_val
                row["RMSE 95% CI"] = ci_str
            elif metric == "mae":
                row["MAE Mean"] = mean_val
                row["MAE 95% CI"] = ci_str
            elif metric == "r2":
                row["R²"] = mean_val
                row["R² 95% CI"] = ci_str
            elif metric == "pearsonr":
                row["Pearson r"] = mean_val
                row["Pearson r 95% CI"] = ci_str
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df[["RMSE Mean", "RMSE 95% CI", "MAE Mean", "MAE 95% CI",
               "R²", "R² 95% CI", "Pearson r", "Pearson r 95% CI"]]


def plot_bootstrap_boxplots(
    task_raw: dict,
    baseline_key: str = None,
    save_path: str = None
) -> None:
    """Side-by-side bootstrap RMSE and R² boxplots across models.

    Args:
        task_raw: Dict mapping model name -> raw bootstrap arrays dict
            (keys: 'rmse', 'mae', 'r2', 'pearsonr').
        baseline_key: Model name whose mean is drawn as a dashed reference line.
        save_path: If provided, save the figure to this path.
    """
    model_names = list(task_raw.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, title in [
        (axes[0], "rmse", "RMSE (years)", "Bootstrap RMSE Distribution (1000 resamples)"),
        (axes[1], "r2",   "R²",           "Bootstrap R² Distribution (1000 resamples)"),
    ]:
        data = [task_raw[m][metric] for m in model_names]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if baseline_key and baseline_key in task_raw:
            ax.axhline(
                np.mean(task_raw[baseline_key][metric]),
                color=colors[0], linestyle="--", linewidth=1,
                label=f"{baseline_key} mean"
            )
            ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_stability_selection(
    selection_counts: dict,
    stable_features: list,
    n_subsamples: int = 50,
    threshold: float = 0.5,
    save_path: str = None
) -> None:
    """Histogram of all feature selection frequencies and ranked bar chart of stable features.

    Args:
        selection_counts: Dict mapping feature name -> selection count.
        stable_features: List of features that passed the stability threshold.
        n_subsamples: Total number of subsamples used (for axis labelling).
        threshold: Stability threshold as a fraction (for threshold line).
        save_path: If provided, save the figure to this path.
    """
    counts = np.array(list(selection_counts.values()))
    min_count = threshold * n_subsamples

    stable_sorted = sorted(
        [(f, selection_counts[f]) for f in stable_features],
        key=lambda x: x[1], reverse=True
    )
    feat_counts = [c for _, c in stable_sorted] if stable_sorted else []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(counts, bins=range(0, n_subsamples + 2), color="#4C72B0",
                 edgecolor="white", linewidth=0.4)
    axes[0].axvline(min_count, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"{int(threshold*100)} % threshold ({int(min_count)})")
    axes[0].set_xlabel(f"Selection frequency (out of {n_subsamples} subsamples)")
    axes[0].set_ylabel("Number of CpG features")
    axes[0].set_title("Stability Selection — Frequency Distribution")
    axes[0].legend()

    axes[1].bar(range(len(feat_counts)), feat_counts, color="#55A868",
                edgecolor="none", width=1.0)
    axes[1].axhline(min_count, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"{int(threshold*100)} % threshold")
    axes[1].set_xlabel("Stable CpG features (ranked by frequency)")
    axes[1].set_ylabel("Selection frequency")
    axes[1].set_title(f"Stable Features (n={len(stable_features)}) — Ranked by Selection Frequency")
    axes[1].set_xticks([])
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def select_mrmr_K(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cpg_features: list,
    K_values: list,
    target_col: str = 'age'
) -> Tuple[int, dict, list]:
    """Sweep K values for mRMR and evaluate validation RMSE at each K.

    Runs mRMR once for max(K_values) (greedy forward selection means the
    first K features are identical for all K ≤ max_K), then fits OLS on
    each prefix to find the elbow in validation RMSE.

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        cpg_features: CpG feature column names.
        K_values: Sorted list of K values to evaluate.
        target_col: Target column name.

    Returns:
        Tuple of (best_K, dict K→val_rmse, all_selected features up to max K).
    """
    from mrmr import mrmr_regression

    X_train = train_df[cpg_features].fillna(train_df[cpg_features].mean())
    y_train = train_df[target_col]
    train_col_means = train_df[cpg_features].mean()
    X_val = val_df[cpg_features].fillna(train_col_means)
    y_val = val_df[target_col]

    max_K = max(K_values)
    all_selected = mrmr_regression(X=X_train, y=y_train, K=max_K, show_progress=False)

    rmse_per_K = {}
    for K in K_values:
        feats = all_selected[:K]
        lr = LinearRegression()
        lr.fit(X_train[feats], y_train)
        y_pred = lr.predict(X_val[feats])
        rmse_per_K[K] = float(np.sqrt(mean_squared_error(y_val, y_pred)))

    best_K = min(rmse_per_K, key=rmse_per_K.get)
    return best_K, rmse_per_K, all_selected


def select_mrmr_features(
    train_df: pd.DataFrame,
    cpg_features: list,
    K: int,
    all_selected: list,
    target_col: str = 'age'
) -> Tuple[list, pd.DataFrame]:
    """Extract top-K mRMR features and compute their Spearman |r| importance scores.

    Args:
        train_df: Training dataframe (used for Spearman computation).
        cpg_features: CpG feature column names.
        K: Number of features to select.
        all_selected: Pre-computed mRMR ordering (from select_mrmr_K).
        target_col: Target column name.

    Returns:
        Tuple of (selected feature names list,
                  DataFrame with columns ['Rank', 'Feature', 'Spearman |r|']).
    """
    X = train_df[cpg_features].fillna(train_df[cpg_features].mean())
    y = train_df[target_col]
    selected = all_selected[:K]
    rows = []
    for rank, feat in enumerate(selected, 1):
        r = abs(spearmanr(X[feat], y).statistic)
        rows.append({'Rank': rank, 'Feature': feat, 'Spearman |r|': round(r, 4)})
    importance_df = pd.DataFrame(rows).set_index('Rank')
    return selected, importance_df


def plot_mrmr_K_curve(
    K_values: list,
    rmse_per_K: dict,
    best_K: int,
    save_path: str = None
) -> None:
    """Line plot of validation RMSE vs K for mRMR feature selection.

    Args:
        K_values: Sorted list of K values evaluated.
        rmse_per_K: Dict mapping K -> validation RMSE.
        best_K: The chosen K (highlighted with a marker).
        save_path: If provided, save figure to this path.
    """
    ks = sorted(K_values)
    rmses = [rmse_per_K[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, rmses, marker='o', color="#4C72B0", linewidth=2, markersize=5)
    ax.axvline(best_K, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Best K = {best_K} (RMSE = {rmse_per_K[best_K]:.3f})")
    ax.set_xlabel("Number of features K")
    ax.set_ylabel("Validation RMSE (years)")
    ax.set_title("mRMR: Validation RMSE vs K")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_set_overlap(
    set1: list,
    set2: list,
    label1: str = "Set 1",
    label2: str = "Set 2",
    title: str = "Feature Set Overlap",
    save_path: str = None
) -> None:
    """Venn-style diagram showing overlap between two feature sets.

    Draws two proportionally-spaced circles annotated with set sizes and
    overlap count, plus a bar chart of the three partition counts.

    Args:
        set1: First feature list (e.g. stability-selected).
        set2: Second feature list (e.g. mRMR-selected).
        label1: Label for set1.
        label2: Label for set2.
        title: Figure title.
        save_path: If provided, save figure to this path.
    """
    s1, s2 = set(set1), set(set2)
    only1 = len(s1 - s2)
    only2 = len(s2 - s1)
    both  = len(s1 & s2)
    total = len(s1 | s2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: Venn diagram using circles ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    c1 = plt.Circle((3.5, 3), 2.4, color="#4C72B0", alpha=0.45)
    c2 = plt.Circle((6.5, 3), 2.4, color="#DD8452", alpha=0.45)
    ax.add_patch(c1)
    ax.add_patch(c2)

    ax.text(2.2, 3, str(only1), ha='center', va='center', fontsize=15, fontweight='bold')
    ax.text(5.0, 3, str(both),  ha='center', va='center', fontsize=15, fontweight='bold')
    ax.text(7.8, 3, str(only2), ha='center', va='center', fontsize=15, fontweight='bold')

    ax.text(2.2, 0.4, f"Only\n{label1}", ha='center', va='center', fontsize=9, color="#4C72B0")
    ax.text(5.0, 0.4, "Both",            ha='center', va='center', fontsize=9, color="grey")
    ax.text(7.8, 0.4, f"Only\n{label2}", ha='center', va='center', fontsize=9, color="#DD8452")

    ax.set_title(title, fontsize=11, pad=4)
    ax.text(5.0, 5.7, f"Union = {total} unique features", ha='center', fontsize=9, color='grey')

    # --- Right: bar chart of partition counts ---
    ax2 = axes[1]
    bars = [only1, both, only2]
    bar_labels = [f"Only\n{label1}", "Both", f"Only\n{label2}"]
    colors = ["#4C72B0", "#8172B3", "#DD8452"]
    ax2.bar(bar_labels, bars, color=colors, alpha=0.8, edgecolor='white')
    for i, v in enumerate(bars):
        ax2.text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')
    ax2.set_ylabel("Number of CpG features")
    ax2.set_title("Partition Sizes")
    ax2.set_xlabel(
        f"Jaccard similarity = {both}/{total} = {both/total:.2f}"
        if total else ""
    )

    plt.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"  {label1} only : {only1}")
    print(f"  Shared        : {both}  ({100*both/len(s1):.1f}% of {label1}, {100*both/len(s2):.1f}% of {label2})")
    print(f"  {label2} only : {only2}")
    print(f"  Jaccard       : {both}/{total} = {both/total:.3f}" if total else "")


def tune_model(
    dev_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    model,
    param_dist: dict,
    n_iter: int = 50,
    cv: int = 5,
    seed: int = 42,
) -> Tuple[object, dict, pd.DataFrame]:
    """Tune model hyperparameters using RandomizedSearchCV on the full development set.

    Cross-validation is performed on dev_df; the best estimator is automatically
    refit on all of dev_df (refit=True). The objective minimised is mean CV RMSE.

    Args:
        dev_df: Full development dataframe (train + validation combined).
        feature_cols: Feature column names (CpG only; no categorical).
        target_col: Target column name.
        model: Unfitted sklearn estimator instance.
        param_dist: Hyperparameter search distributions keyed by bare param name
            (without 'regressor__' prefix — this is added internally).
        n_iter: Number of random search iterations (≥40 required).
        cv: Number of cross-validation folds.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
          - best_pipeline: Pipeline refit on full dev_df with best params.
          - best_params: Dict of best hyperparameter values (bare names).
          - cv_results_df: DataFrame of all trials sorted by CV RMSE.
    """
    preprocessor = create_preprocessor(feature_cols, [])
    pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', model)])

    prefixed_dist = {f'regressor__{k}': v for k, v in param_dist.items()}

    X = dev_df[feature_cols]
    y = dev_df[target_col]

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=prefixed_dist,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X, y)

    best_params = {
        k.replace('regressor__', ''): v
        for k, v in search.best_params_.items()
    }

    cv_df = pd.DataFrame(search.cv_results_).copy()
    cv_df['mean_cv_rmse'] = -cv_df['mean_test_score']
    cv_df['std_cv_rmse']  =  cv_df['std_test_score']
    cv_df = (
        cv_df[['rank_test_score', 'mean_cv_rmse', 'std_cv_rmse', 'params']]
        .sort_values('rank_test_score')
        .reset_index(drop=True)
    )
    cv_df['params'] = cv_df['params'].apply(
        lambda d: {k.replace('regressor__', ''): v for k, v in d.items()}
    )

    return search.best_estimator_, best_params, cv_df


def plot_tuning_cv_results(
    cv_results: dict,
    save_path: str = None,
) -> None:
    """Box plot of CV RMSE distributions across all RandomizedSearchCV trials.

    Args:
        cv_results: Dict mapping model name -> cv_results_df (from tune_model).
        save_path: If provided, save figure to this path.
    """
    model_names = list(cv_results.keys())
    data = [cv_results[m]['mean_cv_rmse'].values for m in model_names]
    colors = ["#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Highlight the best (min mean RMSE) per model
    for i, name in enumerate(model_names):
        best_rmse = cv_results[name]['mean_cv_rmse'].min()
        ax.scatter(i + 1, best_rmse, marker='*', s=120, color='gold',
                   zorder=5, label=f'{name} best: {best_rmse:.3f}' if i == 0 else f'{name}: {best_rmse:.3f}')

    ax.set_xticks(range(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, rotation=10, ha='right')
    ax.set_ylabel('Mean CV RMSE (years)')
    ax.set_title('RandomizedSearchCV — CV RMSE distribution across 50 trials')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_all_metrics_boxplots(
    results: dict,
    stage: str = "FS+Tuned",
    save_path: str = None,
) -> None:
    """2×2 grid of bootstrap boxplots (RMSE, MAE, R², Pearson r) for tuned models.

    Args:
        results: Nested dict results[stage][model] = (summary, raw_arrays).
        stage: Which stage to plot (typically 'FS+Tuned').
        save_path: If provided, save figure to this path.
    """
    model_names = list(results[stage].keys())
    colors = ["#DD8452", "#55A868", "#C44E52"]
    metrics = [
        ("rmse",     "RMSE (years)",     "RMSE"),
        ("mae",      "MAE (years)",       "MAE"),
        ("r2",       "R²",                "R²"),
        ("pearsonr", "Pearson r",         "Pearson r"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (key, ylabel, title) in zip(axes, metrics):
        data = [results[stage][m][1][key] for m in model_names]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=10, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {stage} models (1000 bootstrap resamples)")

    plt.suptitle(f"Bootstrap metric distributions — {stage}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pred_vs_actual(
    pipelines: dict,
    eval_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'age',
    best_model: str = None,
    save_path: str = None,
) -> None:
    """Scatter plots of Predicted vs Actual age for each tuned model.

    Args:
        pipelines: Dict mapping model name -> fitted pipeline.
        eval_df: Evaluation dataframe.
        feature_cols: Feature column names shared by all pipelines.
        target_col: Target column name.
        best_model: If provided, that panel gets a gold border.
        save_path: If provided, save figure to this path.
    """
    model_names = list(pipelines.keys())
    n = len(model_names)
    colors = ["#DD8452", "#55A868", "#C44E52"]

    y_true = eval_df[target_col].values
    age_range = np.array([y_true.min() - 2, y_true.max() + 2])

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name, color in zip(axes, model_names, colors):
        y_pred = pipelines[name].predict(eval_df[feature_cols])
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        r, _ = pearsonr(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.65, color=color, edgecolors='white',
                   linewidths=0.4, s=55)
        ax.plot(age_range, age_range, 'k--', linewidth=1, label='Perfect prediction')

        # Regression line through the predictions
        m_coef = np.polyfit(y_true, y_pred, 1)
        ax.plot(age_range, np.polyval(m_coef, age_range),
                color=color, linewidth=1.5, linestyle='-', alpha=0.8)

        ax.set_xlabel('Actual Age (years)')
        ax.set_ylabel('Predicted Age (years)')
        ax.set_title(f'{name}\nRMSE={rmse:.3f}  R²={r2:.3f}  r={r:.3f}')
        ax.set_xlim(age_range)
        ax.set_ylim(age_range)
        ax.legend(fontsize=8)

        if name == best_model:
            for spine in ax.spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(3)

    plt.suptitle('Predicted vs Actual Age — FS+Tuned models (evaluation set)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def optuna_tune_model(
    model_name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv: int = 5,
    seed: int = 42,
) -> Tuple[object, dict, list]:
    """Tune model hyperparameters using Optuna TPE (Tree-structured Parzen Estimator).

    Unlike random search, TPE builds a probabilistic surrogate of the objective
    and samples more densely from promising regions identified in earlier trials.

    The search spaces mirror those used in tune_model / RandomizedSearchCV:
      ElasticNet   : alpha log-uniform [0.001, 10], l1_ratio uniform [0.1, 1.0]
      SVR          : C log-uniform [0.1, 500], epsilon {0.01,0.1,0.5,1.0},
                     kernel {rbf, linear}
      BayesianRidge: alpha_1/2, lambda_1/2 log-uniform [1e-7, 1e-3]

    Args:
        model_name: One of 'ElasticNet', 'SVR', 'BayesianRidge'.
        pipeline: Template sklearn Pipeline (preprocessor + default regressor).
            Each trial clones this and sets new regressor hyperparameters.
        X_train: Feature DataFrame used for cross-validation.
        y_train: Target Series.
        n_trials: Number of Optuna trials (>=50 recommended).
        cv: Number of cross-validation folds.
        seed: Random seed for the TPE sampler.

    Returns:
        Tuple of:
          - best_pipeline: Pipeline refit on all of X_train with best params.
          - best_params: Dict of best hyperparameter values (bare names).
          - trial_rmses: List of mean CV RMSE for each trial in order.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_name == 'ElasticNet':
            params = {
                'regressor__alpha':    trial.suggest_float('alpha',    0.001, 10,   log=True),
                'regressor__l1_ratio': trial.suggest_float('l1_ratio', 0.1,   1.0),
            }
        elif model_name == 'SVR':
            params = {
                'regressor__C':       trial.suggest_float('C', 0.1, 500, log=True),
                'regressor__epsilon': trial.suggest_categorical('epsilon', [0.01, 0.1, 0.5, 1.0]),
                'regressor__kernel':  trial.suggest_categorical('kernel',  ['rbf', 'linear']),
            }
        elif model_name == 'BayesianRidge':
            params = {
                'regressor__alpha_1':  trial.suggest_float('alpha_1',  1e-7, 1e-3, log=True),
                'regressor__alpha_2':  trial.suggest_float('alpha_2',  1e-7, 1e-3, log=True),
                'regressor__lambda_1': trial.suggest_float('lambda_1', 1e-7, 1e-3, log=True),
                'regressor__lambda_2': trial.suggest_float('lambda_2', 1e-7, 1e-3, log=True),
            }
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        pipe = clone(pipeline)
        pipe.set_params(**params)
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        return float(-scores.mean())

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_prefixed = {f'regressor__{k}': v for k, v in best_params.items()}
    best_pipe = clone(pipeline)
    best_pipe.set_params(**best_prefixed)
    best_pipe.fit(X_train, y_train)

    trial_rmses = [t.value for t in study.trials]
    return best_pipe, best_params, trial_rmses


def plot_optuna_history(
    trial_rmses_dict: dict,
    highlight_model: str = None,
    save_path: str = None,
) -> None:
    """Plot Optuna optimisation history: trial RMSE vs trial number.

    Shows the per-trial RMSE and the running best (cumulative minimum) for
    each model. Helps visualise TPE convergence speed.

    Args:
        trial_rmses_dict: Dict mapping model name -> list of per-trial RMSE.
        highlight_model: If given, that model's panel gets a gold border.
        save_path: If provided, save figure to this path.
    """
    model_names = list(trial_rmses_dict.keys())
    n = len(model_names)
    colors = ["#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name, color in zip(axes, model_names, colors):
        rmses = trial_rmses_dict[name]
        trials = range(1, len(rmses) + 1)
        running_best = np.minimum.accumulate(rmses)

        ax.scatter(trials, rmses, alpha=0.4, color=color, s=20, label='Trial RMSE')
        ax.plot(trials, running_best, color=color, linewidth=2,
                label='Running best', zorder=3)
        ax.axhline(running_best[-1], color='black', linestyle='--', linewidth=1,
                   label=f'Best: {running_best[-1]:.4f}')
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Mean CV RMSE (years)')
        ax.set_title(name + (' ★' if name == highlight_model else ''))
        ax.legend(fontsize=8)

        if name == highlight_model:
            for spine in ax.spines.values():
                spine.set_edgecolor('gold')
                spine.set_linewidth(2.5)

    plt.suptitle('Optuna TPE Optimisation History (50 trials, 5-fold CV)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def select_sex_mrmr(
    train_df: pd.DataFrame,
    cpg_features: list,
    K: int,
    target_col: str = 'sex_encoded',
    seed: int = 42,
) -> list:
    """Select K CpG features for sex classification using mRMR (classif variant).

    Args:
        train_df: Training dataframe with encoded sex column (0/1).
        cpg_features: All candidate CpG feature names.
        K: Number of features to select.
        target_col: Name of the binary target column.
        seed: Random seed (passed to show_progress=False; mrmr_classif is deterministic).

    Returns:
        List of K selected feature names in mRMR order.
    """
    from mrmr import mrmr_classif
    X = train_df[cpg_features].fillna(train_df[cpg_features].mean())
    y = train_df[target_col]
    selected = mrmr_classif(X=X, y=y, K=K, show_progress=False)
    return selected


def bootstrap_classify(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    return_raw: bool = False,
) -> dict:
    """Bootstrap evaluation of a pre-fitted classification pipeline.

    Args:
        pipeline: Already-fitted sklearn classifier Pipeline.
        X: Feature array for evaluation.
        y: Binary target array (0/1 encoded).
        n_bootstraps: Number of bootstrap resamples.
        seed: Random seed.
        return_raw: If True, include raw bootstrap arrays under key 'raw'.

    Returns:
        Dict with keys:
            '{metric}_mean': float — bootstrap mean
            '{metric}_ci':   (lo, hi) tuple — 95% percentile CI
        Metrics: accuracy, f1, mcc, roc_auc, pr_auc.
        If return_raw=True, also includes 'raw': dict of 1-D arrays.
    """
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_arr  = np.array(y)

    np.random.seed(seed)
    raw = {'accuracy': [], 'f1': [], 'mcc': [], 'roc_auc': [], 'pr_auc': []}
    for _ in range(n_bootstraps):
        idx = np.random.choice(len(y_arr), len(y_arr), replace=True)
        ys, yp, yprob = y_arr[idx], y_pred[idx], y_prob[idx]
        raw['accuracy'].append(accuracy_score(ys, yp))
        raw['f1'].append(f1_score(ys, yp, zero_division=0))
        raw['mcc'].append(matthews_corrcoef(ys, yp))
        # Guard against degenerate bootstrap samples with only one class
        if len(np.unique(ys)) > 1:
            raw['roc_auc'].append(roc_auc_score(ys, yprob))
            raw['pr_auc'].append(average_precision_score(ys, yprob))
        else:
            raw['roc_auc'].append(np.nan)
            raw['pr_auc'].append(np.nan)

    results = {}
    for metric_name, values in raw.items():
        arr = np.array(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        mean  = float(np.mean(valid))
        lo    = float(np.percentile(valid, 2.5))
        hi    = float(np.percentile(valid, 97.5))
        results[f'{metric_name}_mean'] = mean
        results[f'{metric_name}_ci']   = (lo, hi)

    if return_raw:
        results['raw'] = {k: np.array(v, dtype=float) for k, v in raw.items()}
    return results


def plot_sex_cpg_bar(
    train_df: pd.DataFrame,
    sex_features: list,
    target_col: str = 'sex_encoded',
    top_n: int = 20,
    save_path: str = None,
) -> pd.DataFrame:
    """Bar chart of top CpG features by |point-biserial r| with sex.

    Args:
        train_df: Training dataframe with encoded sex column.
        sex_features: mRMR-selected CpG feature names (defines which probes to rank).
        target_col: Binary sex column name.
        top_n: Number of probes to display.
        save_path: If provided, save figure to this path.

    Returns:
        DataFrame of top probes with their point-biserial r values.
    """
    y = train_df[target_col].values
    rows = []
    for feat in sex_features:
        x = train_df[feat].fillna(train_df[feat].mean()).values
        r, p = pointbiserialr(y, x)
        rows.append({'CpG probe': feat, 'Point-biserial r': r, '|r|': abs(r), 'p-value': p})
    pb_df = (pd.DataFrame(rows)
               .sort_values('|r|', ascending=False)
               .head(top_n)
               .reset_index(drop=True))
    pb_df.index += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#C44E52' if r > 0 else '#4C72B0' for r in pb_df['Point-biserial r']]
    ax.barh(pb_df['CpG probe'][::-1], pb_df['Point-biserial r'][::-1],
            color=colors[::-1], edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Point-biserial r  (positive → higher β in males)')
    ax.set_title(f'Top {top_n} sex-discriminative CpG probes by |point-biserial r|')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return pb_df


def plot_confusion_matrices(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    class_names: list = None,
    save_path: str = None,
) -> None:
    """Side-by-side confusion matrices for each classifier.

    Args:
        pipelines: Dict mapping model name -> fitted pipeline.
        X: Feature DataFrame.
        y: True labels.
        class_names: Display names for classes (default ['F (0)', 'M (1)']).
        save_path: If provided, save figure to this path.
    """
    if class_names is None:
        class_names = ['F (0)', 'M (1)']

    n = len(pipelines)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, pipe) in zip(axes, pipelines.items()):
        y_pred = pipe.predict(X)
        cm = confusion_matrix(y, y_pred)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(name)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[i, j] > thresh else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Confusion Matrices — Evaluation Set', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    save_path: str = None,
) -> None:
    """ROC curves for all classifiers on the same axes, with AUC annotations.

    Args:
        pipelines: Dict mapping model name -> fitted pipeline.
        X: Feature DataFrame.
        y: True binary labels.
        save_path: If provided, save figure to this path.
    """
    colors = ['#DD8452', '#55A868', '#C44E52', '#4C72B0']
    fig, ax = plt.subplots(figsize=(6, 5))

    for (name, pipe), color in zip(pipelines.items(), colors):
        y_prob = pipe.predict_proba(X)[:, 1]
        RocCurveDisplay.from_predictions(
            y, y_prob, name=name, ax=ax, color=color
        )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
    ax.set_title('ROC Curves — Evaluation Set')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_classification_boxplots(
    raw_results: dict,
    save_path: str = None,
) -> None:
    """2×2 bootstrap boxplots for Accuracy, F1, MCC, ROC-AUC across classifiers.

    Args:
        raw_results: Dict mapping model name -> raw arrays dict
            (keys: 'accuracy', 'f1', 'mcc', 'roc_auc').
        save_path: If provided, save figure to this path.
    """
    model_names = list(raw_results.keys())
    colors = ['#DD8452', '#55A868', '#C44E52', '#4C72B0'][:len(model_names)]
    metrics = [
        ('accuracy', 'Accuracy',   'Accuracy'),
        ('f1',       'F1 Score',   'F1'),
        ('mcc',      'MCC',        'MCC'),
        ('roc_auc',  'ROC-AUC',    'ROC-AUC'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax, (key, ylabel, title) in zip(axes, metrics):
        data = [raw_results[m][key] for m in model_names]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=10, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} (1000 bootstrap resamples)')

    plt.suptitle('Bootstrap Classification Metrics — Evaluation Set',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
