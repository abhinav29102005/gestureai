#!/usr/bin/env python3
"""
Complete CNN-LSTM Model for Human Activity & Gesture Recognition
Uses WISDM dataset (phone/smartwatch accelerometer + gyroscope data)
Includes preprocessing, training, evaluation, XAI (SHAP), and quantization
"""

import os
import sys
import glob
import pickle
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
import joblib
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
WINDOW_SIZE = 128
STRIDE = 64
BATCH_SIZE = 128
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Activity key mapping
ACTIVITY_KEY = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}


def load_wisdm_data() -> pd.DataFrame:
    """Load WISDM dataset from raw_data folder with phone accel + gyro data"""
    print("[*] Loading WISDM dataset...")

    raw_data_path = Path("raw_data/raw/phone")

    if not raw_data_path.exists():
        raise FileNotFoundError(f"raw_data not found at {raw_data_path}")

    # Load accelerometer data
    accel_files = glob.glob(str(raw_data_path / "accel" / "*.txt"))
    accel_dfs = []

    for accel_file in sorted(accel_files):
        try:
            df = pd.read_csv(
                accel_file,
                header=None,
                names=["user_id", "activity", "timestamp", "x_acc", "y_acc", "z_acc"],
            )
            accel_dfs.append(df)
        except Exception as e:
            print(f"[!] Error loading {accel_file}: {e}")

    accel_data = pd.concat(accel_dfs, ignore_index=True)
    print(f"[+] Loaded {len(accel_data)} accelerometer readings")

    # Load gyroscope data and merge
    gyro_files = glob.glob(str(raw_data_path / "gyro" / "*.txt"))
    gyro_dfs = []

    for gyro_file in sorted(gyro_files):
        try:
            df = pd.read_csv(
                gyro_file,
                header=None,
                names=["user_id", "activity", "timestamp", "x_gyro", "y_gyro", "z_gyro"],
            )
            gyro_dfs.append(df)
        except Exception as e:
            print(f"[!] Error loading {gyro_file}: {e}")

    if gyro_dfs:
        gyro_data = pd.concat(gyro_dfs, ignore_index=True)
        # Merge on user_id, activity, timestamp to align accel & gyro
        data = pd.merge(
            accel_data,
            gyro_data,
            on=["user_id", "activity", "timestamp"],
            how="inner",
        )
        print(f"[+] Merged with {len(data)} gyroscope readings")
    else:
        # Use only accelerometer if gyro unavailable
        data = accel_data
        data["x_gyro"] = 0.0
        data["y_gyro"] = 0.0
        data["z_gyro"] = 0.0
        print("[*] Gyroscope data not found, using accelerometer only")

    # Map activity labels from letter codes to names
    data["activity_name"] = data["activity"].map(ACTIVITY_KEY)
    data = data.dropna(subset=["activity_name"])

    print(f"[+] Total dataset size: {len(data)} samples from {data['user_id'].nunique()} subjects")
    print(f"[+] Activity classes: {sorted(data['activity_name'].unique())}")

    return data


def create_sliding_windows(
    X: np.ndarray, y: np.ndarray, user_ids: np.ndarray, window_size: int, stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows with specified stride (50% overlap when stride=window_size/2)"""
    windows_X = []
    windows_y = []
    windows_user = []

    # Group by user to avoid cross-subject windows
    unique_users = np.unique(user_ids)

    for user in unique_users:
        user_mask = user_ids == user
        user_X = X[user_mask]
        user_y = y[user_mask]

        # Create windows for this user's data
        for start_idx in range(0, len(user_X) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = user_X[start_idx:end_idx]
            # Use most common label in window for majority voting
            label = np.bincount(user_y[start_idx:end_idx]).argmax()

            windows_X.append(window)
            windows_y.append(label)
            windows_user.append(user)

    X_windows = np.array(windows_X)
    y_windows = np.array(windows_y)
    user_windows = np.array(windows_user)

    print(f"[+] Created {len(X_windows)} windows from {len(X)} samples")

    return X_windows, y_windows, user_windows


def preprocess_data(
    data: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder, np.ndarray]:
    """Preprocess: normalize, create windows, encode labels"""
    print("[*] Preprocessing data...")

    # Encode activity labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(data["activity_name"].values)

    # Extract sensor features (6 channels: 3 accel + 3 gyro)
    sensor_cols = ["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]
    X = data[sensor_cols].values
    user_ids = data["user_id"].values

    # Create sliding windows
    X_windows, y_windows, user_windows = create_sliding_windows(
        X, y_encoded, user_ids, WINDOW_SIZE, STRIDE
    )

    # Normalize each window independently using fitted scaler (fit only on training data later)
    scaler = StandardScaler()

    # For now, just reshape for stratified split
    X_windows_reshaped = X_windows.reshape(-1, 6)

    print(f"[+] Label classes: {le.classes_}")
    print(f"[+] Windows shape before split: {X_windows.shape}")

    return X_windows, y_windows, user_windows, scaler, le


def stratified_split_by_subject(
    X: np.ndarray, y: np.ndarray, user_ids: np.ndarray, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split train/val/test by subject to prevent data leakage"""
    print("[*] Performing stratified split by subject...")

    unique_users = np.unique(user_ids)
    # Ensure stratification by selecting diverse users for each split
    train_users, temp_users = train_test_split(
        unique_users, test_size=test_size, random_state=42
    )
    val_users, test_users = train_test_split(
        temp_users, test_size=0.5, random_state=42
    )

    train_mask = np.isin(user_ids, train_users)
    val_mask = np.isin(user_ids, val_users)
    test_mask = np.isin(user_ids, test_users)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"[+] Train: {len(X_train)} windows from {len(train_users)} subjects")
    print(f"[+] Val: {len(X_val)} windows from {len(val_users)} subjects")
    print(f"[+] Test: {len(X_test)} windows from {len(test_users)} subjects")

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_and_handle_imbalance(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on training data, normalize all splits, apply SMOTE"""
    print("[*] Normalizing data and handling imbalance...")

    # Fit scaler on training data only (z-score normalization per channel)
    scaler = StandardScaler()
    # Reshape for fitting: (num_windows * window_size, num_channels)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)

    # Normalize all splits
    def normalize_windows(X, scaler):
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = scaler.transform(X_reshaped)
        return X_normalized.reshape(X.shape)

    X_train_norm = normalize_windows(X_train, scaler)
    X_val_norm = normalize_windows(X_val, scaler)
    X_test_norm = normalize_windows(X_test, scaler)

    # Apply SMOTE on training windows only
    X_train_reshaped = X_train_norm.reshape(X_train_norm.shape[0], -1)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_reshaped, y_train)
    X_train_final = X_train_smote.reshape(-1, WINDOW_SIZE, 6)

    print(f"[+] After SMOTE: {len(X_train_final)} training windows")
    print(f"[+] Training class distribution: {np.bincount(y_train_smote)}")

    return X_train_final, X_val_norm, X_test_norm, y_train_smote, scaler


class TemporalAttention(nn.Module):
    """Temporal self-attention layer: compute attention weights over timesteps"""

    def __init__(self, hidden_size: int):
        super().__init__()
        # Linear layer to compute attention scores for each timestep
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch_size, seq_len, hidden_size)
        Returns:
            context: (batch_size, hidden_size) - weighted sum of LSTM outputs
            attention_weights: (batch_size, seq_len) - normalized attention scores
        """
        # Compute attention scores: (batch_size, seq_len, 1)
        scores = self.attention(lstm_out)
        # Softmax over timesteps: (batch_size, seq_len)
        attention_weights = torch.softmax(scores.squeeze(-1), dim=1)
        # Weighted sum: (batch_size, hidden_size)
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context, attention_weights


class CNNLSTMModel(nn.Module):
    """CNN-LSTM with temporal attention for activity recognition"""

    def __init__(self, num_classes: int, input_channels: int = 6):
        super().__init__()

        # CNN block 1: Conv1d(6, 64, kernel=5) + BatchNorm + ReLU
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        # CNN block 2: Conv1d(64, 128, kernel=3) + BatchNorm + ReLU + MaxPool + Dropout
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = nn.Dropout(0.3)

        # LSTM: 2 layers, 128->256 hidden units, dropout between layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
        )

        # Temporal attention layer
        self.attention = TemporalAttention(256)

        # Classifier head: FC(256->128) + ReLU + Dropout + FC(128->num_classes)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout_fc = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (batch_size, seq_len, input_channels)
            return_attention: if True, return attention weights
        Returns:
            logits: (batch_size, num_classes)
            attention_weights (optional): (batch_size, seq_len)
        """
        # CNN processing: input (B, L, 6) -> permute to (B, 6, L)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # Permute back to (B, L', 128) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM: returns (output, (h_n, c_n))
        lstm_out, _ = self.lstm(x)

        # Temporal attention: compute weighted sum of LSTM outputs
        context, attention_weights = self.attention(lstm_out)

        # Classifier head
        x = torch.relu(self.fc1(context))
        x = self.dropout_fc(x)
        logits = self.fc2(x)

        if return_attention:
            return logits, attention_weights
        return logits


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced dataset (inverse frequency)"""
    class_counts = np.bincount(y, minlength=num_classes)
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    y_train: np.ndarray,
) -> Tuple[nn.Module, Dict]:
    """Train CNN-LSTM model with early stopping and LR scheduling"""
    print("[*] Training model...")

    # Weighted loss for imbalanced classes
    class_weights = compute_class_weights(y_train, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=LR_SCHEDULER_PATIENCE, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(logits, 1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"[+] Epoch {epoch + 1}/{MAX_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} [BEST]"
                )
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(
                    f"[+] Epoch {epoch + 1}/{MAX_EPOCHS} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
                )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"[*] Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth", map_location=DEVICE))
    return model, history


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> Dict:
    """Evaluate model on test set, compute metrics"""
    print("[*] Evaluating model on test set...")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.softmax(logits, dim=1)
        y_pred = torch.argmax(probs, dim=1).cpu().numpy()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[+] Test Accuracy: {accuracy:.4f}")
    print(f"[+] Macro F1: {macro_f1:.4f}")
    print(f"[+] Weighted F1: {weighted_f1:.4f}")

    # Classification report
    class_report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        digits=4,
    )
    print("\n[+] Classification Report:")
    print(class_report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_pred": y_pred,
        "probs": probs.cpu().numpy(),
        "cm": cm,
        "class_report": class_report,
    }

    return metrics


def plot_training_history(history: Dict):
    """Plot training and validation loss/accuracy curves"""
    print("[*] Plotting training history...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(history["val_acc"], label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_training_history.png", dpi=300, bbox_inches="tight")
    print("[+] Saved: 01_training_history.png")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, le: LabelEncoder):
    """Plot 18x18 confusion matrix heatmap"""
    print("[*] Plotting confusion matrix...")

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Activity", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Activity", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix - Activity Recognition", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("[+] Saved: 02_confusion_matrix.png")
    plt.close()


def plot_tsne_embeddings(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
):
    """Plot t-SNE of LSTM hidden states colored by activity class"""
    print("[*] Computing t-SNE embeddings...")

    # Extract LSTM outputs (before FC layer)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        # Get LSTM output and attention weights
        x = X_test_tensor.permute(0, 2, 1)
        x = torch.relu(model.bn1(model.conv1(x)))
        x = torch.relu(model.bn2(model.conv2(x)))
        x = model.pool(x)
        x = model.dropout_cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = model.lstm(x)
        context, _ = model.attention(lstm_out)

    embeddings = context.cpu().numpy()

    # Compute t-SNE with limited samples for speed
    sample_indices = np.random.choice(len(embeddings), size=min(2000, len(embeddings)), replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings[sample_indices])
    y_sample = y_test[sample_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=y_sample,
        cmap="tab20",
        s=50,
        alpha=0.6,
        edgecolors="k",
        linewidth=0.5,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight="bold")
    ax.set_title("t-SNE of LSTM Hidden States by Activity Class", fontsize=14, fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Activity Class", fontsize=11)
    cbar.set_ticks(range(len(le.classes_)))
    cbar.set_ticklabels(le.classes_, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_tsne_embeddings.png", dpi=300, bbox_inches="tight")
    print("[+] Saved: 03_tsne_embeddings.png")
    plt.close()


def plot_attention_heatmaps(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
    num_samples: int = 5,
):
    """Plot attention weights heatmap for sample windows"""
    print("[*] Plotting attention weight heatmaps...")

    # Select diverse samples
    sample_indices = np.random.choice(len(X_test), size=num_samples, replace=False)

    X_samples = torch.tensor(X_test[sample_indices], dtype=torch.float32, device=DEVICE)
    y_samples = y_test[sample_indices]

    model.eval()
    with torch.no_grad():
        _, attention_weights = model(X_samples, return_attention=True)

    attention_weights = attention_weights.cpu().numpy()

    # Plot heatmap: rows = samples, columns = timesteps
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(attention_weights, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Timestep", fontsize=12, fontweight="bold")
    ax.set_ylabel("Window Sample", fontsize=12, fontweight="bold")
    ax.set_title("Attention Weights - Which Timesteps Does Model Focus On?", fontsize=14, fontweight="bold")
    ax.set_yticks(range(num_samples))
    ax.set_yticklabels([f"{le.classes_[y_samples[i]]} (#{sample_indices[i]})" for i in range(num_samples)])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_attention_heatmaps.png", dpi=300, bbox_inches="tight")
    print("[+] Saved: 04_attention_heatmaps.png")
    plt.close()


def compute_shap_values(
    model: nn.Module,
    X_test: np.ndarray,
    le: LabelEncoder,
    num_samples: int = 200,
):
    """Compute SHAP values for input sensor channels"""
    print("[*] Computing SHAP values...")

    # Use subset for SHAP (it's computationally expensive)
    sample_indices = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
    X_sample = X_test[sample_indices]

    X_sample_tensor = torch.tensor(X_sample, dtype=torch.float32, device=DEVICE)

    # Create wrapper for SHAP
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            logits = model(x_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    # Use background data for SHAP
    background_indices = np.random.choice(len(X_sample), size=min(50, len(X_sample)), replace=False)
    background = X_sample[background_indices]

    explainer = shap.GradientExplainer(model, torch.tensor(background, dtype=torch.float32, device=DEVICE))
    shap_values = explainer.shap_values(X_sample_tensor)

    # Average across samples and classes
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)

    # Reshape: (num_classes, num_samples, window_size, 6)
    # We want mean absolute SHAP per channel
    if len(shap_values.shape) == 4:
        # Average over samples and timesteps
        mean_shap = np.abs(shap_values).mean(axis=(1, 2))  # (num_classes, 6)
    else:
        # If 3D: (num_samples, window_size, 6)
        mean_shap = np.abs(shap_values).mean(axis=(0, 1))  # (6,)
        mean_shap = np.tile(mean_shap, (len(le.classes_), 1))

    return mean_shap, shap_values


def plot_shap_importance(mean_shap: np.ndarray, le: LabelEncoder):
    """Plot mean absolute SHAP per sensor axis"""
    print("[*] Plotting SHAP importance...")

    sensor_names = ["X Accel", "Y Accel", "Z Accel", "X Gyro", "Y Gyro", "Z Gyro"]

    # Overall importance (mean across all classes)
    overall_importance = mean_shap.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Bar chart: overall importance
    axes[0].barh(sensor_names, overall_importance, color="steelblue")
    axes[0].set_xlabel("Mean Absolute SHAP Value", fontsize=12, fontweight="bold")
    axes[0].set_title("Overall Sensor Importance (All Activities)", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="x")

    # Heatmap: per-activity importance
    im = axes[1].imshow(mean_shap.T, cmap="RdYlBu_r", aspect="auto")
    axes[1].set_xlabel("Activity Class", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Sensor", fontsize=12, fontweight="bold")
    axes[1].set_title("SHAP Importance by Activity", fontsize=14, fontweight="bold")
    axes[1].set_xticks(range(len(le.classes_)))
    axes[1].set_xticklabels(le.classes_, rotation=45, ha="right")
    axes[1].set_yticks(range(len(sensor_names)))
    axes[1].set_yticklabels(sensor_names)
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label("Mean Absolute SHAP", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_shap_importance.png", dpi=300, bbox_inches="tight")
    print("[+] Saved: 05_shap_importance.png")
    plt.close()

    # Save SHAP values for dashboard
    np.save(OUTPUT_DIR / "shap_values.npy", mean_shap)


def quantize_model(model: nn.Module) -> nn.Module:
    """Apply INT8 dynamic quantization to Linear layers"""
    print("[*] Quantizing model with INT8...")

    # Get model size before quantization
    model_size_before = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # MB

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    # Get model size after quantization
    model_size_after = sum(p.numel() for p in quantized_model.parameters()) * 4 / (1024 * 1024)  # approximate

    print(f"[+] Model size before: {model_size_before:.2f} MB")
    print(f"[+] Model size after: {model_size_after:.2f} MB")

    return quantized_model, model_size_before, model_size_after


def benchmark_inference_latency(model: nn.Module, model_quant: nn.Module):
    """Benchmark inference latency before and after quantization"""
    print("[*] Benchmarking inference latency...")

    # Create dummy batch
    X_dummy = torch.randn(100, WINDOW_SIZE, 6, device=DEVICE)

    # Original model
    model.eval()
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_dummy)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    time_original = (time.time() - start) / 10 * 1000  # ms

    # Quantized model
    model_quant.eval()
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model_quant(X_dummy)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    time_quant = (time.time() - start) / 10 * 1000  # ms

    print(f"[+] Original model latency (100-sample batch): {time_original:.2f} ms")
    print(f"[+] Quantized model latency (100-sample batch): {time_quant:.2f} ms")
    print(f"[+] Speedup: {time_original / time_quant:.2f}x")

    return time_original, time_quant


def main():
    """Main pipeline: load, preprocess, train, evaluate, optimize"""
    print("=" * 80)
    print("CNN-LSTM Activity Recognition - WISDM Dataset")
    print("=" * 80)

    # 1. Load data
    data = load_wisdm_data()

    # 2. Preprocess: normalize, create windows, encode labels
    X_windows, y_windows, user_windows, scaler_init, le = preprocess_data(data)

    # 3. Stratified split by subject
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_by_subject(
        X_windows, y_windows, user_windows
    )

    # 4. Normalize and handle imbalance
    X_train, X_val, X_test, y_train, scaler = normalize_and_handle_imbalance(
        X_train, X_val, X_test, y_train, y_val
    )

    # 5. Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Build and train model
    model = CNNLSTMModel(num_classes=len(le.classes_)).to(DEVICE)
    model, history = train_model(model, train_loader, val_loader, len(le.classes_), y_train)

    # 7. Evaluate
    metrics = evaluate_model(model, X_test, y_test, le)

    # 8. Generate visualizations
    plot_training_history(history)
    plot_confusion_matrix(metrics["cm"], le)
    plot_tsne_embeddings(model, X_test, y_test, le)
    plot_attention_heatmaps(model, X_test, y_test, le)
    mean_shap, _ = compute_shap_values(model, X_test, le)
    plot_shap_importance(mean_shap, le)

    # 9. Model optimization
    model_quant, model_size_before, model_size_after = quantize_model(model)
    time_original, time_quant = benchmark_inference_latency(model, model_quant)

    # 10. Save artifacts
    print("\n[*] Saving artifacts...")
    joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
    print("[+] Saved: label_encoder.pkl, scaler.pkl, best_model.pth")

    # 11. Summary report
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Macro F1 Score:              {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score:           {metrics['weighted_f1']:.4f}")
    print(f"Test Accuracy:               {metrics['accuracy']:.4f}")
    print(f"\nModel Size (Before):         {model_size_before:.2f} MB")
    print(f"Model Size (After INT8):     {model_size_after:.2f} MB")
    print(f"\nInference Latency (Original): {time_original:.2f} ms (100-sample batch)")
    print(f"Inference Latency (Quantized): {time_quant:.2f} ms (100-sample batch)")
    print(f"Speedup:                     {time_original / time_quant:.2f}x")
    print(f"\nAll outputs saved to:        {OUTPUT_DIR.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
