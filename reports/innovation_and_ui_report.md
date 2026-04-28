# Innovation & UI Design Report: GesturAI Suite

## 1. Architectural Innovations
- **Multi-Scale Convolutional Backbone**: Unlike standard single-kernel CNNs, we use a parallel 3-kernel branch (3, 5, 11). This captures multi-frequency sensor patterns — high-frequency jitters (Typing, Clapping), medium-scale movements (Walking, Jogging), and large slow cycles (Stairs, Folding Clothes) — essential for the 18-class WISDM v2 dataset.
- **Temporal Attention Mechanism**: A learned attention layer scans each 200-sample window to focus on the specific milliseconds where gesture transitions occur, ignoring background noise and static periods.
- **Temporal Attenuation (Signal Decay)**: An exponential decay kernel `λ_t = e^(-γ(T-t))` ensures recent motion samples dominate predictions, improving gesture boundary detection over uniform weighting.
- **8-Axis Fusion Confidence Score**: Standard softmax confidence is often overconfident. Our entropy-penalized fusion score measures inter-sensor agreement — if the 8 channels disagree (high attribution entropy), confidence is automatically reduced, providing industrial operators a trustworthy certainty metric.

## 2. Industrial Optimizations
- **Zero-Latency Playback Mode**: A pre-computation engine (`precompute_results.py`) exports all inference and XAI results to Parquet files (51 subjects × ~650 windows each = 34,326 total frames). This allows **instant demonstration** of the full dataset without any GPU/CPU lag at runtime.
- **High-Throughput DataFrame Streaming**: The API uses Pure DataFrame operations with Parquet-backed storage for sub-millisecond data retrieval and O(1) subject indexing.
- **ONNX Export**: The trained model is exported to ONNX format for low-latency edge deployment (<15ms inference on standard CPU).
- **SMOTE Balancing**: Synthetic Minority Over-sampling ensures all 18 activity classes are equally represented during training, preventing bias toward frequent activities like Sitting (2,096 windows) over rare ones like Typing (1,719 windows).

## 3. UI/UX Professionalism
- **Industrial Dark Theme**: The dashboard uses a high-contrast dark theme (#0b0e14 background) with the **Outfit** typeface for maximum readability in industrial environments.
- **4-Tab Architecture**: Live Analytics, Subject Profile, Theory & Science, and Global Intelligence — each serving a distinct operational need.
- **Per-Subject Profiling**: Selecting any of the 51 subjects displays their average confidence, total inference windows, top activity, activity distribution pie chart, gesture clarity histogram, and temporal attention patterns.
- **Global Intelligence Dashboard**: Displays system-wide KPIs — 87.9% average confidence across all subjects, 34,326 analyzed windows, and global sensor attribution breakdown.
- **Mathematical Transparency**: Integrated LaTeX rendering directly into the Theory tab, covering all 7 mathematical foundations from Resultant Magnitude through AdamW optimization.
- **Real-Time Sensor Visualization**: 8-axis signal fusion plots, confidence gauges, attention heatmaps, and gradient attribution bars update live during analysis playback.

## 4. Dataset & Model Summary
| Metric | Value |
| :--- | :--- |
| Total Records | 3,440,324 |
| Subjects | 51 (IDs 1600–1650) |
| Activities | 18 |
| Inference Frames | 34,326 |
| Sensor Axes | 8 (Acc XYZ + Res, Gyro XYZ + Res) |
| Window Size | 200 samples |
| Stride | 100 samples |
| Avg Prediction Confidence | 87.9% |
| Best Activity Confidence | 92.0% (Dribbling) |
| Lowest Activity Confidence | 81.0% (Eating Sandwich) |

## 5. Improvements Over Initial Scope
- **Subject-Stratified Splitting**: Zero data leakage by splitting at the subject level (80/20), an industrial requirement often missed in standard ML projects.
- **XAI Integration**: Real-time Local Gradient Attribution allows users to see exactly which of the 8 sensors drives each prediction — enabling true Glass Box transparency.
- **Entropy-Penalized Confidence**: Moves beyond naive softmax to a fusion score that penalizes sensor disagreement, giving operators actionable certainty metrics.
- **Precomputed Results Pipeline**: All 51 subjects are batch-processed with full XAI (SHAP + Gradients + Attention), stored as Parquet for instant dashboard loading.
