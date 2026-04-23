import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ARCHITECTURE ---
class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(in_channels, out_channels//3, 3, padding=1), nn.BatchNorm1d(out_channels//3), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv1d(in_channels, out_channels//3, 5, padding=2), nn.BatchNorm1d(out_channels//3), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv1d(in_channels, out_channels//3 + out_channels%3, 11, padding=5), nn.BatchNorm1d(out_channels//3 + out_channels%3), nn.ReLU())
    def forward(self, x): return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)

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
        self.res_link = nn.Conv1d(8, 128, 1)
        self.lstm = nn.LSTM(128, 256, 2, batch_first=True, dropout=0.5)
        self.attn = TemporalAttention(256)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))
    def forward(self, x, return_attn=False):
        identity = self.res_link(x.transpose(1, 2))
        x = self.ms_conv1(x.transpose(1, 2))
        x = self.ms_conv2(x)
        x = self.pool(x + identity[:, :, :x.shape[2]])
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        ctx, w = self.attn(out)
        logits = self.fc(ctx)
        return (logits, w) if return_attn else logits

def precompute():
    os.makedirs("data/results", exist_ok=True)
    logger.info("Loading assets...")
    le = joblib.load('data/processed/label_encoder.pkl')
    scaler = joblib.load('data/processed/scaler.pkl')
    model = SOTA_CNNLSTM(len(le.classes_))
    model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
    model.eval()
    
    logger.info("Loading dataset...")
    df = pd.read_csv("data/processed/processed_full_dataset.csv")
    sensors = ['x_acc', 'y_acc', 'z_acc', 'res_acc', 'x_gyro', 'y_gyro', 'z_gyro', 'res_gyro']
    
    WINDOW_SIZE = 200
    STRIDE = 100
    
    subjects = df['user_id'].unique()
    for sub in tqdm(subjects, desc="Precomputing Subjects"):
        sub_df = df[df['user_id'] == sub].copy()
        sub_df[sensors] = scaler.transform(sub_df[sensors])
        data = sub_df[sensors].values
        
        results = []
        for i in range(0, len(data) - WINDOW_SIZE, STRIDE):
            window = data[i:i+WINDOW_SIZE]
            X = torch.FloatTensor(window).unsqueeze(0)
            X.requires_grad = True
            
            out, attns = model(X, return_attn=True)
            probs = torch.softmax(out, dim=1).detach().numpy()
            idx = np.argmax(probs[0])
            
            # XAI Gradients
            model.zero_grad()
            out[0, idx].backward()
            grads = torch.abs(X.grad).mean(dim=1).squeeze().detach().numpy()
            local_imp = (grads / (grads.sum() + 1e-8)).tolist()
            
            results.append({
                "activity": le.classes_[idx],
                "confidence": float(probs[0][idx]),
                "attention": attns.squeeze().detach().numpy().tolist(),
                "local_imp": local_imp,
                "raw_window": window.tolist() # Optional: store raw for playback signals
            })
            
        with open(f"data/results/sub_{sub}.json", "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    precompute()
