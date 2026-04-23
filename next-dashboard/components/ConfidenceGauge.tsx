"use client";
import { motion } from "framer-motion";

interface ConfidenceGaugeProps {
  value: number; // 0-1
  size?: number;
}

export default function ConfidenceGauge({ value, size = 140 }: ConfidenceGaugeProps) {
  const radius = 52;
  const cx = size / 2;
  const cy = size / 2 + 12;
  const startAngle = -210;
  const sweepAngle = 240;
  const pct = Math.min(Math.max(value, 0), 1);

  function polarToCart(angle: number, r: number) {
    const rad = (angle * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  }

  function describeArc(startDeg: number, endDeg: number, r: number) {
    const s = polarToCart(startDeg, r);
    const e = polarToCart(endDeg, r);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
  }

  const fillEnd = startAngle + sweepAngle * pct;
  const color = pct > 0.8 ? "#34d399" : pct > 0.5 ? "#fbbf24" : "#ef4444";

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size * 0.75} viewBox={`0 0 ${size} ${size * 0.75}`}>
        {/* Track */}
        <path d={describeArc(startAngle, startAngle + sweepAngle, radius)} fill="none" stroke="#1e2d45" strokeWidth="10" strokeLinecap="round" />
        {/* Fill */}
        <motion.path
          d={describeArc(startAngle, fillEnd, radius)}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: pct }}
          transition={{ duration: 1, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
        {/* Needle dot */}
        {(() => {
          const pt = polarToCart(fillEnd, radius);
          return (
            <motion.circle
              cx={pt.x} cy={pt.y} r={5}
              fill={color}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.8 }}
              style={{ filter: `drop-shadow(0 0 8px ${color})` }}
            />
          );
        })()}
        {/* Value */}
        <text x={cx} y={cy + 6} textAnchor="middle" fill={color} fontSize="20" fontWeight="900" fontFamily="Inter, sans-serif">
          {(pct * 100).toFixed(0)}%
        </text>
        <text x={cx} y={cy + 20} textAnchor="middle" fill="#475569" fontSize="8" fontFamily="Inter, sans-serif">
          CONFIDENCE
        </text>
      </svg>
    </div>
  );
}
