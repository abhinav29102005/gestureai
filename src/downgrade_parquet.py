import os
import pandas as pd
import glob

print("Downgrading Parquet files to version 1.0 for Next.js compatibility...")
files = glob.glob("data/results/*.parquet")

for f in files:
    df = pd.read_parquet(f)
    # Overwrite in data/results
    df.to_parquet(f, engine='pyarrow', compression='snappy', version='1.0')
    
    # Overwrite in next-dashboard
    basename = os.path.basename(f)
    next_path = f"next-dashboard/public/data/results/{basename}"
    if os.path.exists(next_path):
        df.to_parquet(next_path, engine='pyarrow', compression='snappy', version='1.0')

print("✅ Successfully downgraded all Parquet files to version 1.0.")
