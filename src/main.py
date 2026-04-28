import os
import glob
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import torch.onnx

# --- INDUSTRIAL LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "data/raw/wisdm-dataset"
RAW_DIR = os.path.join(DATASET_ROOT, "raw", "phone")
CACHE_FILE = "data/processed/data_cache.joblib"
OUTPUT_DIR = "data/outputs"
WINDOW_SIZE = 200 
STRIDE = 100 
BATCH_SIZE = 128
EPOCHS = 40 # Reduced to prevent overfitting
LR = 1e-3

# --- DATA PIPELINE ---
def load_and_preprocess():
    if os.path.exists(CACHE_FILE):
        logger.info("Loading preprocessed data from cache...")
        return joblib.load(CACHE_FILE)

    logger.info("Starting industrial-scale data loading (Watch Only)...")
    def read_sensor_type(sensor_type):
        path = os.path.join(DATASET_ROOT, "raw", "watch", sensor_type)
        all_files = glob.glob(os.path.join(path, "*.txt"))
        dfs = []
        for f in all_files:
            try:
                temp = pd.read_csv(f, header=None, names=['user_id', 'activity', 'timestamp', 'x', 'y', 'z'], engine='c', on_bad_lines='skip')
                temp['z'] = temp['z'].astype(str).str.replace(';', '').astype(float)
                temp['res'] = np.sqrt(temp['x']**2 + temp['y']**2 + temp['z']**2)
                dfs.append(temp)
            except Exception as e:
                logger.warning(f"Skipping corrupt file {f}: {e}")
        return pd.concat(dfs, ignore_index=True)

    acc_df = read_sensor_type("accel")
    gyro_df = read_sensor_type("gyro")
    
    acc_df = acc_df.sort_values(['user_id', 'activity', 'timestamp'])
    gyro_df = gyro_df.sort_values(['user_id', 'activity', 'timestamp'])
    acc_df['idx'] = acc_df.groupby(['user_id', 'activity']).cumcount()
    gyro_df['idx'] = gyro_df.groupby(['user_id', 'activity']).cumcount()
    
    merged = pd.merge(acc_df, gyro_df, on=['user_id', 'activity', 'idx'], suffixes=('_acc', '_gyro'))
    
    activity_map = {
        'A': 'Walking', 'B': 'Jogging', 'C': 'Stairs', 'D': 'Sitting', 'E': 'Standing',
        'F': 'Typing', 'G': 'Brushing Teeth', 'H': 'Eating Soup', 'I': 'Eating Chips', 'J': 'Eating Pasta',
        'K': 'Drinking', 'L': 'Eating Sandwich', 'M': 'Kicking', 'O': 'Playing Catch', 'P': 'Dribbling',
        'Q': 'Writing', 'R': 'Clapping', 'S': 'Folding Clothes'
    }
    merged['activity_name'] = merged['activity'].map(activity_map)
    le = LabelEncoder()
    merged['activity_idx'] = le.fit_transform(merged['activity_name'])
    
    sensors = ['x_acc', 'y_acc', 'z_acc', 'res_acc', 'x_gyro', 'y_gyro', 'z_gyro', 'res_gyro']
    subjects = merged['user_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    train_subs = subjects[:int(0.8*len(subjects))]
    test_subs = subjects[int(0.8*len(subjects)):]
    
    train_df = merged[merged['user_id'].isin(train_subs)]
    test_df = merged[merged['user_id'].isin(test_subs)]
    
    scaler = StandardScaler()
    train_df[sensors] = scaler.fit_transform(train_df[sensors])
    test_df[sensors] = scaler.transform(test_df[sensors])
    
    def get_windows(data):
        X, y = [], []
        for _, g in data.groupby(['user_id', 'activity_name']):
            f = g[sensors].values
            l = g['activity_idx'].values
            for i in range(0, len(f) - WINDOW_SIZE, STRIDE):
                X.append(f[i:i+WINDOW_SIZE])
                y.append(np.bincount(l[i:i+WINDOW_SIZE]).argmax())
        return np.array(X), np.array(y)

    X_tr, y_tr = get_windows(train_df)
    X_te, y_te = get_windows(test_df)
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_tr.reshape(len(X_tr), -1), y_tr)
    X_tr = X_res.reshape(-1, WINDOW_SIZE, 8)
    y_tr = y_res
    
    result = (X_tr, y_tr, X_te, y_te, le, scaler, sensors)
    joblib.dump(result, CACHE_FILE)
    return result

from model import SOTA_CNNLSTM

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    X_tr, y_tr, X_te, y_te, le, scaler, sensors = load_and_preprocess()
    joblib.dump(le, 'data/processed/label_encoder.pkl')
    joblib.dump(scaler, 'data/processed/scaler.pkl')
    
    train_ld = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)), BATCH_SIZE, True)
    test_ld = DataLoader(TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)), BATCH_SIZE, False)
    
    model = SOTA_CNNLSTM(len(le.classes_)).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2) # Increased weight decay for regularization
    crit = nn.CrossEntropyLoss(label_smoothing=0.1) # Added label smoothing to prevent overfitting
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) 
    
    logger.info("Starting High-Performance Training Loop...")
    best_acc = 0
    patience = 10
    patience_counter = 0
    for e in range(EPOCHS):
        model.train()
        t_loss = 0
        train_corr = 0
        train_total = 0
        for X, y in train_ld:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(X)
            l = crit(out, y)
            l.backward(); opt.step()
            t_loss += l.item()
            train_corr += (out.argmax(1) == y).sum().item()
            train_total += len(y)
        
        train_acc = train_corr / train_total
        sched.step()
        model.eval()
        corr = 0
        with torch.no_grad():
            for X, y in test_ld:
                corr += (model(X.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
        acc = corr / len(X_te)
        
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
            logger.info(f"🔥 New Best Val Accuracy: {best_acc:.4f} (Train Acc: {train_acc:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {e} with validation accuracy {best_acc:.4f}")
            break
        
        if e % 5 == 0:
            logger.info(f"Epoch {e} | Loss: {t_loss/len(train_ld):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {acc:.4f}")

    # Final Eval
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X, _ in test_ld:
            y_pred.extend(model(X.to(DEVICE)).argmax(1).cpu().numpy())
    
    report = classification_report(y_te, y_pred, target_names=le.classes_)
    logger.info("\n" + report)
    
    # Robust ONNX Export
    try:
        logger.info("Exporting optimized ONNX graph natively to opset 18...")
        dummy = torch.randn(1, WINDOW_SIZE, 8).to(DEVICE)
        torch.onnx.export(
            model, 
            dummy, 
            "models/best_model.onnx", 
            input_names=['sensor_input'], 
            output_names=['activity_output'], 
            opset_version=18,
            export_params=True,
            do_constant_folding=True
        )
    except Exception as e:
        logger.error(f"ONNX Export failed: {e}. Dashboard will use fallback PyTorch model.")

    logger.info("High-Performance Pipeline Complete.")

if __name__ == "__main__":
    main()
