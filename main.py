from src.data_preprocessing import preprocess_dataset


def main():
    train_df, test_df = preprocess_dataset(raw_path="data/raw/heart.csv")
    print(f"Processed train dataset with {len(train_df)} rows")
    print(f"Processed test dataset with {len(test_df)} rows")


if __name__ == "__main__":
    main()
