from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(path_str: str = "data/raw/heart.csv") -> pd.DataFrame:
    """Load the raw dataset from disk."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned dataset with missing values checked. Just a safety check for now."""
    df = df.copy()
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values. Please handle them before continuing.")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features using one-hot encoding for the heart dataset."""
    df = df.copy()
    categorical_cols = ["cp", "restecg", "slope", "thal"]
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
    return df


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame | None = None,
    scaler: StandardScaler = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    """Scale continuous numeric feature columns with a train-only fit."""
    X_train = X_train.copy()
    if scaler is None:
        scaler = StandardScaler()

    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test_scaled, scaler


def split_features_target(df: pd.DataFrame, target_column: str = "target") -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/cleaned_data.csv") -> None:
    """Save a processed dataset to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train/test sets with reproducible shuffling."""
    split_args = {
        "test_size": test_size,
        "random_state": random_state,
    }
    if stratify:
        split_args["stratify"] = y
    return train_test_split(X, y, **split_args)


def preprocess_dataset(
    raw_path: str = "data/raw/heart.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, encode, split, scale, and save train/test processed datasets."""
    df = load_dataset(raw_path)
    df = clean_dataset(df)
    df = encode_features(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state, stratify=True
    )
    X_train, X_test, scaler = scale_features(X_train, X_test)

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_processed_data(train_df, output_dir / "train.csv")
    save_processed_data(test_df, output_dir / "test.csv")

    import joblib
    joblib.dump(scaler, output_dir / "scaler.pkl")
    
    return train_df, test_df


def preview_dataset(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return a preview of the dataset."""
    return df.head(n)
