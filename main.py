from pathlib import Path

import pandas as pd

from src.baseline_model import (
    train_logistic_regression,
    evaluate_model,
    train_mlp,
    evaluate_mlp,
    save_logistic_regression,
    save_mlp,
)
from src.data_preprocessing import preprocess_dataset, split_features_target


def load_data(raw_path="data/raw/heart.csv", processed_dir="data/processed"):
    """Load and preprocess data if needed."""
    train_path = Path(processed_dir) / "train.csv"
    test_path = Path(processed_dir) / "test.csv"

    if not train_path.exists() or not test_path.exists():
        print("[INFO] Preprocessing dataset...")
        preprocess_dataset(raw_path=raw_path, output_dir=processed_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def prepare_features_and_targets(train_df, test_df):
    """Split data into features and targets."""
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    return X_train, y_train, X_test, y_test


def train_and_evaluate_baseline_models(X_train, y_train, X_test, y_test):
    """Train and evaluate both Logistic Regression and MLP."""
    results = {}

    # Logistic Regression
    print("\n[TRAIN] Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    results["logistic_regression"] = lr_metrics
    print(f"[RESULTS] Logistic Regression - Accuracy: {lr_metrics['accuracy']:.4f}, AUC: {lr_metrics['auc']:.4f}")
    
    # Save Logistic Regression
    lr_save_path = save_logistic_regression(lr_model)
    print(f"[SAVED] Logistic Regression model at {lr_save_path}")

    # MLP
    print("\n[TRAIN] MLP (50 epochs)...")
    input_dim = X_train.shape[1]
    mlp_model, losses = train_mlp(X_train, y_train, input_dim=input_dim, epochs=50)
    mlp_metrics = evaluate_mlp(mlp_model, X_test, y_test)
    results["mlp"] = mlp_metrics
    print(f"[RESULTS] MLP - Accuracy: {mlp_metrics['accuracy']:.4f}, AUC: {mlp_metrics['auc']:.4f}")
    
    # Save MLP
    mlp_save_path = save_mlp(mlp_model)
    print(f"[SAVED] MLP model at {mlp_save_path}")

    return results


def main():
    """Main training pipeline."""
    print("[START] Federated Health ML - Baseline Models")

    # Load data
    train_df, test_df = load_data()
    print(f"[INFO] Loaded train: {len(train_df)} rows, test: {len(test_df)} rows")

    # Prepare features and targets
    X_train, y_train, X_test, y_test = prepare_features_and_targets(train_df, test_df)
    print(f"[INFO] Features: {X_train.shape[1]}")

    # Train and evaluate baseline models
    results = train_and_evaluate_baseline_models(X_train, y_train, X_test, y_test)

    # Summary
    print("\n[SUMMARY]")
    for model_name, metrics in results.items():
        print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
