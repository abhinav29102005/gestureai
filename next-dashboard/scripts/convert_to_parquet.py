import os
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = "public/data/results"

def convert_to_parquet():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found.")
        return

    # Find all unique subject IDs
    subject_ids = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith(".json") and f.startswith("sub_"):
            sid = f.split("_p")[0].replace("sub_", "").replace(".json", "")
            subject_ids.add(sid)
    
    print(f"Found {len(subject_ids)} subjects. Converting to Parquet...")

    for sid in sorted(list(subject_ids)):
        print(f"  Converting Subject {sid}...")
        sub_items = []
        
        # Collect all JSON chunks
        single_file = os.path.join(DATA_DIR, f"sub_{sid}.json")
        chunks = []
        if os.path.exists(single_file):
            chunks.append(single_file)
        else:
            chunk_idx = 0
            while True:
                chunk_path = os.path.join(DATA_DIR, f"sub_{sid}_p{chunk_idx}.json")
                if not os.path.exists(chunk_path):
                    break
                chunks.append(chunk_path)
                chunk_idx += 1
        
        for chunk in chunks:
            with open(chunk, 'r') as f:
                sub_items.extend(json.load(f))
        
        if not sub_items:
            print(f"    No data found for subject {sid}")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(sub_items)
        
        # Ensure correct types for list columns
        # Pandas might represent them as objects/lists, which pyarrow can convert
        
        output_path = os.path.join(DATA_DIR, f"sub_{sid}.parquet")
        
        # Use Parquet version 1.0 for compatibility with parquetjs-lite
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, version='1.0', compression='none')
        
        print(f"    Created {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
        
        # Cleanup JSON files
        for chunk in chunks:
            os.remove(chunk)
            print(f"    Removed {chunk}")

    print("Conversion complete.")

if __name__ == "__main__":
    convert_to_parquet()
