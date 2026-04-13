from src.data_preprocessing import load_dataset


def main():
    data = load_dataset()
    print(f"Loaded dataset with {len(data)} rows")


if __name__ == "__main__":
    main()
