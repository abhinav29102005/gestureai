import torch
import torch.nn as nn

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
        self.ms_conv1 = MultiScaleCNN(8, 128)
        self.ms_conv2 = MultiScaleCNN(128, 256)
        self.pool = nn.MaxPool1d(2)
        self.res_link = nn.Conv1d(8, 256, 1) # Match dimensions for residual link
        
        self.lstm = nn.LSTM(256, 256, 2, batch_first=True, dropout=0.5) # Reduced hidden size, increased dropout
        self.attn = TemporalAttention(256)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), # Increased dropout
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
