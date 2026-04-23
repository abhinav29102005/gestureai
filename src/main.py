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
EPOCHS = 80 # Increased for better convergence
LR = 1e-3

# --- DATA PIPELINE ---
def load_and_preprocess():
    if os.path.exists(CACHE_FILE):
        logger.info("Loading preprocessed data from cache...")
        return joblib.load(CACHE_FILE)

    logger.info("Starting industrial-scale data loading...")
    def read_sensor_type(sensor_type):
        path = os.path.join(RAW_DIR, sensor_type)
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
        'A': 'walking', 'B': 'jogging', 'C': 'stairs', 'D': 'sitting', 'E': 'standing',
        'F': 'typing', 'G': 'teeth', 'H': 'soup', 'I': 'chips', 'J': 'pasta',
        'K': 'drinking', 'L': 'sandwich', 'M': 'kicking', 'O': 'catch', 'P': 'dribbling',
        'Q': 'writing', 'R': 'clapping', 'S': 'folding'
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

# --- SOTA ARCHITECTURE (Multi-Scale ResNet CNN-LSTM) ---
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(in_channels, out_channels//3, 3, padding=1), nn.BatchNorm1d(out_channels//3), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//3, 5, padding=2), nn.BatchNorm1d(out_channels//3), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//3 + out_channels%3, 11, padding=5), nn.BatchNorm1d(out_channels//3 + out_channels%3), nn.ReLU())
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1), w

class SOTA_CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ms_conv1 = MultiScaleCNN(8, 64)
        self.ms_conv2 = MultiScaleCNN(64, 128)
        self.pool = nn.MaxPool1d(2)
        self.res_link = nn.Conv1d(8, 128, 1) # Match dimensions for residual link
        
        self.lstm = nn.LSTM(128, 256, 2, batch_first=True, dropout=0.5)
        self.attn = TemporalAttention(256)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, return_attn=False):
        identity = self.res_link(x.transpose(1, 2))
        x = self.ms_conv1(x.transpose(1, 2))
        x = self.ms_conv2(x)
        x = self.pool(x + identity[:, :, :x.shape[2]]) # Residual connection (downsampled)
        
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        ctx, w = self.attn(out)
        logits = self.fc(ctx)
        return (logits, w, ctx) if return_attn else logits

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
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # AdamW for better regularization
    crit = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) # Smooth decay
    
    logger.info("Starting High-Performance Training Loop...")
    best_acc = 0
    for e in range(EPOCHS):
        model.train()
        t_loss = 0
        for X, y in train_ld:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l = crit(model(X), y)
            l.backward(); opt.step()
            t_loss += l.item()
        
        sched.step()
        model.eval()
        corr = 0
        with torch.no_grad():
            for X, y in test_ld:
                corr += (model(X.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
        acc = corr / len(X_te)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            logger.info(f"🔥 New Best Accuracy: {best_acc:.4f}")
        
        if e % 5 == 0:
            logger.info(f"Epoch {e} | Loss: {t_loss/len(train_ld):.4f} | Val Acc: {acc:.4f}")

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
        logger.info("Exporting optimized ONNX graph...")
        dummy = torch.randn(1, WINDOW_SIZE, 8).to(DEVICE)
        # Use simpler export mode to avoid onnxscript conflicts if possible
        torch.onnx.export(model, dummy, "models/best_model.onnx", input_names=['sensor_input'], output_names=['activity_output'], opset_version=12)
    except Exception as e:
        logger.error(f"ONNX Export failed: {e}. Dashboard will use fallback PyTorch model.")

    logger.info("High-Performance Pipeline Complete.")

if __name__ == "__main__":
    main()
