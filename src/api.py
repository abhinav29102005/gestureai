import os
import pandas as pd
import numpy as np
import logging
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- GLOBAL DATA STORE ---
FULL_DF = None
SUBJECT_INDICES = {} # Stores (start_idx, end_idx) for each subject
SUBJECT_POINTERS = {} # Stores current relative pointer

def load_industrial_data():
    global FULL_DF, SUBJECT_INDICES, SUBJECT_POINTERS
    logger.info("🚀 API: Initializing High-Throughput DataFrame Streamer...")
    
    csv_path = "data/processed/processed_full_dataset.csv"
    if not os.path.exists(csv_path):
        logger.error(f"Critical Error: {csv_path} not found! Please run training/export first.")
        return False

    try:
        # Load entire CSV into memory as a DataFrame (most memory efficient format)
        logger.info("Loading 3.6M+ records into memory...")
        FULL_DF = pd.read_csv(csv_path)
        
        # Pre-calculate the slice indices for every subject to avoid searching during streaming
        logger.info("Indexing subjects for O(1) retrieval speed...")
        FULL_DF = FULL_DF.sort_values('user_id').reset_index(drop=True)
        
        unique_subs = FULL_DF['user_id'].unique()
        for sub in unique_subs:
            sub_mask = FULL_DF['user_id'] == sub
            indices = FULL_DF.index[sub_mask]
            SUBJECT_INDICES[str(sub)] = (indices.min(), indices.max())
            SUBJECT_POINTERS[str(sub)] = 0
            
        logger.info(f"Ready. Streamer indexed {len(unique_subs)} subjects.")
        return True
    except Exception as e:
        logger.error(f"Streamer Initialization Failed: {e}")
        return False

@app.route('/subjects', methods=['GET'])
def get_subjects():
    return jsonify(list(SUBJECT_INDICES.keys()))

@app.route('/stream_batch/<subject_id>/<int:size>', methods=['GET'])
def stream_batch(subject_id, size):
    if subject_id not in SUBJECT_INDICES: return jsonify({"error": "404"}), 404
    start, end = SUBJECT_INDICES[subject_id]
    ptr = SUBJECT_POINTERS[subject_id]
    
    current_idx = start + ptr
    if current_idx > end: return jsonify({"error": "EOF"}), 400
    
    # Slice the DataFrame for the batch
    batch_size = min(size, end - current_idx + 1)
    batch_df = FULL_DF.iloc[current_idx : current_idx + batch_size].copy()
    batch_df = batch_df.rename(columns={'activity_name': 'true_activity'})
    
    SUBJECT_POINTERS[subject_id] += batch_size
    return jsonify(batch_df.to_dict('records'))

@app.route('/stream/<subject_id>', methods=['GET'])
def stream_data(subject_id):
    if subject_id not in SUBJECT_INDICES:
        return jsonify({"error": "Subject not found"}), 404
    
    start, end = SUBJECT_INDICES[subject_id]
    ptr = SUBJECT_POINTERS[subject_id]
    
    current_idx = start + ptr
    if current_idx > end:
        return jsonify({"error": "EOF"}), 400
    
    # Efficiently extract a single row as a dict
    row = FULL_DF.iloc[current_idx].to_dict()
    # Clean up key names for the dashboard
    row['true_activity'] = row.pop('activity_name')
    
    SUBJECT_POINTERS[subject_id] += 1
    return jsonify(row)

@app.route('/reset/<subject_id>', methods=['GET'])
def reset_p(subject_id):
    if subject_id in SUBJECT_POINTERS:
        SUBJECT_POINTERS[subject_id] = 0
        return jsonify({"status": "reset"})
    return jsonify({"error": "404"}), 404

if __name__ == '__main__':
    if load_industrial_data():
        logger.info("Industrial Simulator live on port 5001.")
        app.run(port=5001, threaded=True)
