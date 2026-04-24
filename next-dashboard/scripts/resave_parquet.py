import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = "public/data/results"

def resave_parquet():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    print(f"Found {len(files)} parquet files. Resaving with version 1.0...")

    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"  Processing {f}...")
        df = pd.read_parquet(path)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, version='1.0', compression='none')
        print(f"    Resaved {f}")

if __name__ == "__main__":
    resave_parquet()
