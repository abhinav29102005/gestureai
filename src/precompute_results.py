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
        self.ms_conv1 = MultiScaleCNN(8, 128)
        self.ms_conv2 = MultiScaleCNN(128, 256)
        self.pool = nn.MaxPool1d(2)
        self.res_link = nn.Conv1d(8, 256, 1)
        self.lstm = nn.LSTM(256, 512, 2, batch_first=True, dropout=0.2)
        self.attn = TemporalAttention(512)
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
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
    os.makedirs("next-dashboard/public/data/results", exist_ok=True)
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
    
    global_activity = {}
    global_imp = [0.0] * 8
    imp_count = 0
    total_frames = 0
    subject_summaries = []
    
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
                "raw_window": window.tolist() 
            })
            
            # Update global stats
            total_frames += 1
            act_name = le.classes_[idx]
            conf = float(probs[0][idx])
            
            if act_name not in global_activity:
                global_activity[act_name] = {"count": 0, "totalConf": 0.0}
            global_activity[act_name]["count"] += 1
            global_activity[act_name]["totalConf"] += conf
            
            for k in range(8):
                global_imp[k] += local_imp[k]
            imp_count += 1
            
        df_results = pd.DataFrame(results)
        df_results.to_parquet(f"data/results/sub_{sub}.parquet", compression='snappy', engine='pyarrow', version='1.0')
        df_results.to_parquet(f"next-dashboard/public/data/results/sub_{sub}.parquet", compression='snappy', engine='pyarrow', version='1.0')
        
        # Subject Summary
        if not df_results.empty:
            sub_act_counts = df_results['activity'].value_counts()
            top_activity = sub_act_counts.index[0] if len(sub_act_counts) > 0 else "unknown"
            avg_conf = df_results['confidence'].mean()
        else:
            top_activity = "unknown"
            avg_conf = 0.0
            
        subject_summaries.append({
            "id": str(sub),
            "totalFrames": len(results),
            "topActivity": top_activity,
            "avgConf": float(avg_conf)
        })

    # Finalize Global Stats
    activity_breakdown = []
    for name, d in global_activity.items():
        activity_breakdown.append({
            "name": name,
            "count": d["count"],
            "avgConf": float(d["totalConf"] / d["count"]) if d["count"] > 0 else 0.0
        })
    activity_breakdown.sort(key=lambda x: x["count"], reverse=True)
    
    avg_sensor_imp = [float(v / imp_count) if imp_count > 0 else 0.0 for v in global_imp]
    
    global_stats = {
        "totalSubjects": len(subjects),
        "totalFrames": total_frames,
        "activityBreakdown": activity_breakdown,
        "avgSensorImp": avg_sensor_imp,
        "subjectSummaries": sorted(subject_summaries, key=lambda x: x["id"])
    }
    
    # Save to data/results
    with open("data/results/global_stats.json", 'w') as f:
        json.dump(global_stats, f)
    with open("data/results/manifest.json", 'w') as f:
        json.dump({"subjects": [str(s) for s in sorted(subjects)]}, f)
        
    # Save to next-dashboard/public/data/results
    with open("next-dashboard/public/data/results/global_stats.json", 'w') as f:
        json.dump(global_stats, f)
    with open("next-dashboard/public/data/results/manifest.json", 'w') as f:
        json.dump({"subjects": [str(s) for s in sorted(subjects)]}, f)
        
    logger.info("Global stats and manifest generated successfully in both directories.")

if __name__ == "__main__":
    precompute()
