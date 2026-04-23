"use client";
import { motion } from "framer-motion";

const IMPACT_COLOR: Record<string, string> = {
  Critical: "#ef4444",
  High:     "#fbbf24",
  Medium:   "#34d399",
};

const IMPACT_RANK: Record<string, number> = { Critical: 0, High: 1, Medium: 2 };

const CATEGORIES = [
  {
    category: "Data Pipeline",
    color: "#00f2fe",
    accent: "rgba(0,242,254,0.08)",
    items: [
      {
        title:  "Replace JSON with Parquet",
        impact: "Critical",
        desc:   "Current per-subject JSON files total ~1.2 GB. Parquet with Snappy compression reduces this to ~180 MB (85% reduction) with faster columnar read times and native streaming support.",
      },
      {
        title:  "Add Ground-Truth Labels",
        impact: "High",
        desc:   "Results only store the predicted activity. Storing the actual ground-truth label enables in-dashboard confusion matrix rendering and per-class error analysis.",
      },
      {
        title:  "Store Top-K Softmax Probabilities",
        impact: "High",
        desc:   "Only the top-1 prediction is persisted. Storing top-3 softmax scores enables uncertainty quantification and near-miss classification visualization.",
      },
    ],
  },
  {
    category: "Model Architecture",
    color: "#a78bfa",
    accent: "rgba(167,139,250,0.08)",
    items: [
      {
        title:  "Bidirectional LSTM",
        impact: "High",
        desc:   "The current LSTM is unidirectional. A Bi-LSTM doubles temporal context by processing the signal both forward and backward, typically yielding +1–2% accuracy on HAR tasks.",
      },
      {
        title:  "Strided Conv over MaxPool",
        impact: "Medium",
        desc:   "MaxPool(2) discards high-frequency signal information. A learned Conv1d(stride=2) preserves gradient signal and improves XAI attribution quality downstream.",
      },
      {
        title:  "Squeeze-and-Excitation Blocks",
        impact: "Medium",
        desc:   "Adding channel-wise SE attention after each MultiScaleCNN allows the network to dynamically learn which of the 8 sensor axes matter most per activity.",
      },
    ],
  },
  {
    category: "Training Strategy",
    color: "#34d399",
    accent: "rgba(52,211,153,0.08)",
    items: [
      {
        title:  "Time-Series Augmentation",
        impact: "High",
        desc:   "No data augmentation is applied. Jitter, magnitude scaling, and window permutation during training significantly improve generalization to unseen users and device orientations.",
      },
      {
        title:  "Focal Loss over Label Smoothing",
        impact: "Medium",
        desc:   "Rare classes (chips, writing) are under-represented. Focal Loss dynamically up-weights hard examples, outperforming static label smoothing for imbalanced HAR datasets.",
      },
      {
        title:  "Optuna Hyperparameter Search",
        impact: "Medium",
        desc:   "LR=1e-3 and hidden_dim=256 are hand-tuned. An automated Optuna sweep over learning rate, LSTM depth, hidden dimension, and dropout rate would find a near-optimal configuration.",
      },
    ],
  },
  {
    category: "XAI & Explainability",
    color: "#fbbf24",
    accent: "rgba(251,191,36,0.08)",
    items: [
      {
        title:  "Integrated Gradients",
        impact: "High",
        desc:   "The current local_imp uses vanilla input-gradient, which is noisy. Integrated Gradients (via Captum) provides a provably complete, noise-free attribution that sums to the actual output delta.",
      },
      {
        title:  "Class-Conditional SHAP",
        impact: "High",
        desc:   "SHAP is only computed globally. Per-class SHAP would reveal that Gyro Z matters only for rotational gestures while Acc Res dominates locomotion — enabling fine-grained trust insights.",
      },
    ],
  },
  {
    category: "Deployment & Serving",
    color: "#f472b6",
    accent: "rgba(244,114,182,0.08)",
    items: [
      {
        title:  "INT8 ONNX Quantization",
        impact: "High",
        desc:   "ONNX export exists but is FP32. INT8 quantization cuts model size by 4x and reduces CPU inference from ~12ms to ~5ms, enabling deployment on low-power edge devices and wearables.",
      },
      {
        title:  "Redis Result Caching",
        impact: "Medium",
        desc:   "The Flask API loads the full CSV into RAM per process. A Redis cache enables shared access across Gunicorn workers, reducing per-worker memory and cold-start time significantly.",
      },
      {
        title:  "Server-Sent Events Streaming",
        impact: "Medium",
        desc:   "Adding an SSE endpoint allows the dashboard to receive real-time frame updates over a persistent HTTP connection, replacing the polling model with true push-based streaming.",
      },
    ],
  },
];

export default function ImprovementsView() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

      {/* Header callout */}
      <div className="insight" style={{ borderLeftColor: "#4f8ef7" }}>
        <strong style={{ color: "#4f8ef7" }}>Automated Pipeline Analysis — </strong>
        The following improvements were identified by auditing{" "}
        <code style={{ color: "#a78bfa", fontSize: "0.8rem" }}>main.py</code>,{" "}
        <code style={{ color: "#a78bfa", fontSize: "0.8rem" }}>precompute_results.py</code>, and the
        statistical patterns across the 51-subject result set. Items are ordered by estimated impact.
      </div>

      {/* Categories */}
      {CATEGORIES.map((cat, ci) => (
        <motion.div key={cat.category} className="card"
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: ci * 0.08 }}>

          {/* Category header */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.25rem" }}>
            <div style={{
              width: 4, height: "1.5rem", borderRadius: 2,
              background: cat.color, flexShrink: 0,
            }} />
            <h2 style={{ fontSize: "1rem", fontWeight: 800, color: cat.color, margin: 0 }}>
              {cat.category}
            </h2>
            <div style={{ flex: 1, height: 1, background: `linear-gradient(90deg, ${cat.color}30, transparent)` }} />
            <span style={{ fontSize: "0.7rem", color: "#475569", fontWeight: 600 }}>
              {cat.items.length} items
            </span>
          </div>

          {/* Items grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "0.875rem" }}>
            {[...cat.items].sort((a, b) => IMPACT_RANK[a.impact] - IMPACT_RANK[b.impact]).map((item, ii) => (
              <motion.div key={item.title}
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: ci * 0.08 + ii * 0.06 }}
                style={{
                  background: cat.accent,
                  border: "1px solid #1e2d45",
                  borderRadius: "0.75rem",
                  padding: "1rem",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.625rem",
                }}>

                {/* Impact badge + title row */}
                <div style={{ display: "flex", alignItems: "flex-start", gap: "0.625rem" }}>
                  <span style={{
                    flexShrink: 0,
                    padding: "0.15rem 0.5rem",
                    borderRadius: "0.3rem",
                    fontSize: "0.65rem",
                    fontWeight: 800,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    background: `${IMPACT_COLOR[item.impact]}20`,
                    color: IMPACT_COLOR[item.impact],
                    border: `1px solid ${IMPACT_COLOR[item.impact]}40`,
                  }}>
                    {item.impact}
                  </span>
                </div>

                <div style={{ fontSize: "0.9rem", fontWeight: 700, color: "#e2e8f0", lineHeight: 1.3 }}>
                  {item.title}
                </div>

                <div style={{ fontSize: "0.78rem", color: "#64748b", lineHeight: 1.7 }}>
                  {item.desc}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      ))}

      {/* Summary row */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
        style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
        {[
          { label: "Critical Issues", count: CATEGORIES.flatMap(c => c.items).filter(i => i.impact === "Critical").length, color: "#ef4444" },
          { label: "High Impact",     count: CATEGORIES.flatMap(c => c.items).filter(i => i.impact === "High").length,     color: "#fbbf24" },
          { label: "Medium Impact",   count: CATEGORIES.flatMap(c => c.items).filter(i => i.impact === "Medium").length,   color: "#34d399" },
        ].map(s => (
          <div key={s.label} className="card" style={{ textAlign: "center", borderColor: `${s.color}30`, boxShadow: `0 0 20px ${s.color}10` }}>
            <div style={{ fontSize: "2.5rem", fontWeight: 900, color: s.color }}>{s.count}</div>
            <div style={{ fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#475569", marginTop: 6 }}>{s.label}</div>
          </div>
        ))}
      </motion.div>
    </div>
  );
}
