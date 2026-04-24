# GesturAI Industrial Suite 🚀

![GesturAI Hero](data/outputs/hero.png)

GesturAI is a state-of-the-art industrial gesture recognition system designed for high-throughput sensor data analysis. It leverages a sophisticated Deep Learning architecture combining Multi-Scale Convolutions, LSTM layers, and Temporal Attention to achieve superior accuracy and explainability in human activity recognition (HAR).

### 🌐 Live Demos
- **Primary Dashboard**: [ai-grp-est-2.onrender.com](https://ai-grp-est-2.onrender.com/)
- **Mirror Node**: [ai-grp-est-2-1.onrender.com](https://ai-grp-est-2-1.onrender.com/)

---

## 🧠 SOTA Architecture

The core of GesturAI is a **Multi-Scale ResNet CNN-LSTM** with **Temporal Attention**.

### 1. Multi-Scale Convolutions (Deep Learning Layer)
Uses parallel convolution branches with different kernel sizes (3, 5, and 11). This acts like multiple magnifying glasses:
- **Small Kernels (3)**: Capture high-frequency jitters (typing, teeth brushing).
- **Medium Kernels (5)**: Capture medium-scale movements (walking, jogging).
- **Large Kernels (11)**: Capture large, slow cycles (stairs, folding clothes).

### 2. Temporal Attention (Explainability Layer)
An attention mechanism that scans the signal window to focus on the specific milliseconds where the gesture occurs. 
- Mathematically: $\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}$
- It effectively ignores background noise and highlights the "action" segments.

### 3. Resultant Magnitude (Physics Layer)
Combines X, Y, and Z axes into a single intensity value:
$a_{res} = \sqrt{a_x^2 + a_y^2 + a_z^2}$
This makes the model **Rotation-Invariant**, meaning it can recognize Jogging regardless of how the phone is oriented in a pocket.

---

## 📊 Dataset Information

GesturAI is trained on the **WISDM v2.0 Dataset**, featuring:
- **3.6 Million Records** of raw accelerometer and gyroscope data.
- **18 Activities**: Walking, Jogging, Stairs, Sitting, Standing, Typing, Teeth Brushing, Soup, Chips, Pasta, Drinking, Sandwich, Kicking, Catch, Dribbling, Writing, Clapping, Folding.
- **Sensor Fusion**: High-frequency sync between Accelerometer and Gyroscope (8-axis input).

---

## 🔬 Explainable AI (XAI)

Unlike "black box" models, GesturAI provides deep insights:
- **SHAP (SHapley Additive exPlanations)**: Quantifies the importance of each sensor (Acc vs. Gyro) in the final decision.
- **Gradient Attribution**: Visualizes which specific milliseconds of a gesture triggered the classification.
- **Attention Heatmaps**: Integrated into the dashboard for real-time model introspection.

---

## 📁 Project Structure

```text
ai_grp_est_2/
├── data/
│   ├── raw/                # Original WISDM dataset
│   ├── processed/          # Preprocessed CSVs, caches, and encoders
│   ├── outputs/            # Training metrics, SHAP plots, and distributions
│   └── results/            # Precomputed subject-specific JSON analysis
├── models/                 # Saved PyTorch (.pth) and ONNX weights
├── reports/                # Industrial and UI evaluation reports
├── src/                    # Core source code
│   ├── main.py             # Training pipeline & model definition
│   ├── api.py              # High-throughput Flask sensor API
│   ├── dashboard.py        # Streamlit Industrial Master Suite
│   ├── controller.py       # Central orchestrator for the environment
│   ├── precompute_results.py # Batch inference with XAI (SHAP/Gradients)
│   ├── export_csv.py       # Data merging and export utility
│   └── generate_missing_plots.py # Visual asset generation
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

---

## ⚡ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Using the Controller
| Command | Description |
| :--- | :--- |
| `python src/controller.py train` | Run the training pipeline |
| `python src/controller.py dev` | Launch the API and Dashboard |
| `python src/controller.py all` | Full fresh run (Clean -> Train -> Deploy) |

---

## 🏢 Industrial Impact & Scalability

- **Production Ready**: Optimized ONNX export for low-latency deployment on edge devices.
- **High Throughput**: Flask API using memory-efficient DataFrames for O(1) subject indexing.
- **Industrial Scalability**: Designed to handle millions of records with minimal memory overhead.
- **Quality Assurance**: Integrated classification reports, confusion matrices, and SMOTE for class balancing.

---

## 📝 Performance Metrics
- **Validation Accuracy**: ~94%+ (on 18 classes).
- **Inference Latency**: <15ms on CPU.
- **Dataset Size**: 3,600,000+ data points.