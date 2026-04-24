import os
import json
import math

DATA_DIR = "public/data/results"
MAX_FILE_SIZE = 24 * 1024 * 1024 # 24MB

def split_json_files():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found.")
        return

    all_subjects = set()
    
    # First, collect all existing subjects from files that might already be split or not
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json") and filename != "manifest.json":
            base = filename.split("_p")[0].replace("sub_", "").replace(".json", "")
            all_subjects.add(base)

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json") or "_p" in filename or filename == "manifest.json":
            continue
            
        filepath = os.path.join(DATA_DIR, filename)
        filesize = os.path.getsize(filepath)
        
        if filesize > MAX_FILE_SIZE:
            print(f"Splitting {filename} ({filesize / 1024 / 1024:.2f} MB)...")
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
            
            if not isinstance(data, list):
                print(f"Skipping {filename}: Not a list.")
                continue
                
            items = data
            num_chunks = math.ceil(filesize / (MAX_FILE_SIZE * 0.8)) # Target 80% of limit
            chunk_size = math.ceil(len(items) / num_chunks)
            
            base_name = filename.replace(".json", "")
            
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(items))
                
                chunk_data = items[start:end]
                
                output_name = f"{base_name}_p{i}.json"
                output_path = os.path.join(DATA_DIR, output_name)
                
                with open(output_path, 'w') as f:
                    json.dump(chunk_data, f)
                
                print(f"  Created {output_name}")
            
            # Delete original file
            os.remove(filepath)
            print(f"  Removed original {filename}")

    # Generate manifest
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump({"subjects": sorted(list(all_subjects))}, f)
    print(f"Generated manifest.json with {len(all_subjects)} subjects.")

if __name__ == "__main__":
    split_json_files()
