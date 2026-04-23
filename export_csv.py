import os
import glob
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "wisdm-dataset"
RAW_DIR = os.path.join(DATASET_ROOT, "raw", "phone")

def export():
    logger.info("Merging dataset for CSV export...")
    def read_sensor_type(sensor_type):
        path = os.path.join(RAW_DIR, sensor_type)
        all_files = glob.glob(os.path.join(path, "*.txt"))
        dfs = []
        for f in all_files:
            temp = pd.read_csv(f, header=None, names=['user_id', 'activity', 'timestamp', 'x', 'y', 'z'], on_bad_lines='skip')
            temp['z'] = temp['z'].astype(str).str.replace(';', '').astype(float)
            temp['res'] = np.sqrt(temp['x']**2 + temp['y']**2 + temp['z']**2)
            dfs.append(temp)
        return pd.concat(dfs, ignore_index=True)

    acc_df = read_sensor_type("accel")
    gyro_df = read_sensor_type("gyro")
    
    acc_df = acc_df.sort_values(['user_id', 'activity', 'timestamp'])
    gyro_df = gyro_df.sort_values(['user_id', 'activity', 'timestamp'])
    acc_df['idx'] = acc_df.groupby(['user_id', 'activity']).cumcount()
    gyro_df['idx'] = gyro_df.groupby(['user_id', 'activity']).cumcount()
    
    merged = pd.merge(acc_df, gyro_df, on=['user_id', 'activity', 'idx'], suffixes=('_acc', '_gyro'))
    
    activity_map = {
        'A': 'walking', 'B': 'jogging', 'C': 'stairs', 'D': 'sitting', 'E': 'standing',
        'F': 'typing', 'G': 'teeth', 'H': 'soup', 'I': 'chips', 'J': 'pasta',
        'K': 'drinking', 'L': 'sandwich', 'M': 'kicking', 'O': 'catch', 'P': 'dribbling',
        'Q': 'writing', 'R': 'clapping', 'S': 'folding'
    }
    merged['activity_name'] = merged['activity'].map(activity_map)
    
    # Save the full merged dataset
    logger.info("Saving processed_full_dataset.csv...")
    merged.to_csv("processed_full_dataset.csv", index=False)
    logger.info("Export Complete.")

if __name__ == "__main__":
    export()
