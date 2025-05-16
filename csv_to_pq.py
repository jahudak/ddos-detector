# Make sure to install the dependencies:
# pip install pandas pyarrow

# Example usage:
# python csv_to_pq.py input.csv output.parquet


import pandas as pd
import argparse

def convert_csv_to_parquet(csv_file_path, parquet_file_path):
    df = pd.read_csv(csv_file_path)
    
    df.to_parquet(parquet_file_path, engine='pyarrow')
    print(f"Converted '{csv_file_path}' to '{parquet_file_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("parquet_path", help="Path to save the output Parquet file")
    
    args = parser.parse_args()
    convert_csv_to_parquet(args.csv_path, args.parquet_path)