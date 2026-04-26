import os
import time

# Resolve the absolute path of the dataset relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(script_dir, "../data/processed/processed_full_dataset.csv")

def count_rows():
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    start_time = time.time()
    print(f"Scanning {filepath} for total row count...")
    
    # We use a pure Python generator to count lines without loading the massive file into RAM
    with open(filepath, 'r', encoding='utf-8') as f:
        # subtract 1 to account for the CSV header
        row_count = sum(1 for _ in f) - 1

    elapsed = time.time() - start_time
    print(f"--------------------------------------------------")
    print(f"✅ Total Data Rows (excluding header): {row_count:,}")
    print(f"⏱️ Time taken: {elapsed:.2f} seconds")
    print(f"--------------------------------------------------")

if __name__ == "__main__":
    count_rows()
