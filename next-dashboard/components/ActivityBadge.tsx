"use client";
import { motion } from "framer-motion";
import { getActivityColor } from "@/lib/types";

interface ActivityBadgeProps {
  activity: string;
  confidence?: number;
  size?: "sm" | "md" | "lg";
}

export default function ActivityBadge({ activity, confidence, size = "md" }: ActivityBadgeProps) {
  const color = getActivityColor(activity);
  const padding = size === "sm" ? "px-2 py-0.5 text-xs" : size === "lg" ? "px-4 py-2 text-base" : "px-3 py-1 text-sm";

  return (
    <motion.span
      key={activity}
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 300 }}
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold ${padding}`}
      style={{ backgroundColor: `${color}18`, color, border: `1px solid ${color}40` }}
    >
      <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: color }} />
      {activity}
      {confidence !== undefined && (
        <span className="opacity-70 text-xs">({(confidence * 100).toFixed(0)}%)</span>
      )}
    </motion.span>
  );
}
