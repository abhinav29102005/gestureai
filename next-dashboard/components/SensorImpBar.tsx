"use client";
import { SENSOR_LABELS } from "@/lib/types";
import { motion } from "framer-motion";

interface SensorBarProps {
  values: number[];
  title?: string;
}

const SENSOR_COLORS = ["#60a5fa","#f472b6","#34d399","#00f2fe","#a78bfa","#fbbf24","#fb923c","#14b8a6"];

export default function SensorImpBar({ values, title = "Sensor Attribution" }: SensorBarProps) {
  const max = Math.max(...values, 0.001);

  return (
    <div>
      {title && <div className="text-xs uppercase tracking-widest mb-3" style={{ color: "#475569" }}>{title}</div>}
      <div className="space-y-2">
        {SENSOR_LABELS.map((label, i) => {
          const pct = (values[i] ?? 0) / max;
          const color = SENSOR_COLORS[i];
          return (
            <div key={label} className="flex items-center gap-2">
              <div className="w-16 text-xs text-right shrink-0" style={{ color: "#64748b" }}>{label}</div>
              <div className="flex-1 h-4 rounded-full overflow-hidden" style={{ background: "#1e2d45" }}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct * 100}%` }}
                  transition={{ duration: 0.8, ease: "easeOut", delay: i * 0.05 }}
                  className="h-full rounded-full"
                  style={{ background: `linear-gradient(90deg, ${color}88, ${color})` }}
                />
              </div>
              <div className="w-10 text-xs shrink-0" style={{ color }}>{(values[i] * 100).toFixed(1)}%</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
