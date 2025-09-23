import pandas as pd

# Path to your parquet file
parquet_path = (
    "Users/justinmcclelland/model-res-avm-master/output/intermediate/pin_leaves.parquet"
)

# Path for the CSV output
csv_path = (
    "Users/justinmcclelland/model-res-avm-master/output/intermediate/pin_leaves.csv"
)


def convert_parquet_to_csv(parquet_path, csv_path):
    # Load the parquet file
    df = pd.read_parquet(parquet_path)

    # Save as CSV
    df.to_csv(csv_path, index=False)

    print(f"Converted {parquet_path} → {csv_path}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    convert_parquet_to_csv(parquet_path, csv_path)
