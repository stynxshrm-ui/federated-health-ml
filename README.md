# Federated Health ML

This repository implements a federated learning project around a heart disease prediction dataset.

## Project goals

- Build a centralized baseline model with clean data preprocessing.
- Implement a federated learning simulation using Flower.
- Compare centralized and federated performance.

## Structure

- `data/raw/` — raw dataset files
- `data/processed/` — cleaned and preprocessed data
- `notebooks/` — analysis and experiment notebooks
- `src/` — data pipeline, baseline model, and FL code
- `models/` — saved model artifacts

## Next steps

1. Add the dataset to `data/raw/`.
2. Explore the data in `notebooks/01_eda.ipynb`.
3. Implement the baseline model in `src/baseline_model.py`.
