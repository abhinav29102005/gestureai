"use client";
import { motion } from "framer-motion";

interface AttentionHeatmapProps {
  values: number[]; // 0-1 normalized
}

export default function AttentionHeatmap({ values }: AttentionHeatmapProps) {
  const max = Math.max(...values, 0.001);
  const normalized = values.map((v) => v / max);

  function heatColor(t: number) {
    // Magma-like: black → purple → orange → yellow
    const stops = [
      [0, [0, 0, 0]],
      [0.25, [60, 9, 108]],
      [0.5, [163, 44, 97]],
      [0.75, [240, 130, 40]],
      [1, [252, 253, 191]],
    ] as [number, number[]][];

    for (let i = 0; i < stops.length - 1; i++) {
      const [t0, c0] = stops[i];
      const [t1, c1] = stops[i + 1];
      if (t >= t0 && t <= t1) {
        const f = (t - t0) / (t1 - t0);
        const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
        const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
        const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
        return `rgb(${r},${g},${b})`;
      }
    }
    return "rgb(252,253,191)";
  }

  return (
    <div>
      <div className="text-xs uppercase tracking-widest mb-2" style={{ color: "#475569" }}>Temporal Attention Map</div>
      <div className="flex gap-0.5 h-10 items-end rounded-lg overflow-hidden">
        {normalized.map((v, i) => (
          <motion.div
            key={i}
            className="flex-1 rounded-sm"
            style={{ height: `${Math.max(v * 100, 8)}%`, backgroundColor: heatColor(v), minWidth: 2 }}
            initial={{ scaleY: 0, originY: 1 }}
            animate={{ scaleY: 1 }}
            transition={{ duration: 0.4, delay: i * 0.01 }}
          />
        ))}
      </div>
      <div className="flex justify-between mt-1 text-xs" style={{ color: "#475569" }}>
        <span>t=0</span>
        <span>t=T</span>
      </div>
    </div>
  );
}
