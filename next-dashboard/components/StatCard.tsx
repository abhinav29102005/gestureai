"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface StatCardProps {
  label: string;
  value: string;
  sub?: string;
  glow?: "cyan" | "blue" | "green" | "purple";
  delay?: number;
}

const COLORS = { cyan: "#00f2fe", blue: "#4f8ef7", green: "#34d399", purple: "#a78bfa" };
const ICONS  = { cyan: "SBJ", blue: "WIN", green: "ACC", purple: "LAT" };

export default function StatCard({ label, value, sub, glow = "cyan", delay = 0 }: StatCardProps) {
  const [show, setShow] = useState(false);
  useEffect(() => { const t = setTimeout(() => setShow(true), delay); return () => clearTimeout(t); }, [delay]);
  const color = COLORS[glow];
  const icon  = ICONS[glow];

  return (
    <motion.div className={`stat-card glow-${glow}`}
      initial={{ opacity: 0, y: 18 }} animate={show ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.45, ease: "easeOut" }}>
      <div className="scan-line" />
      <div className="stat-icon-row">
        <span style={{ fontSize: "0.6rem", fontWeight: 800, letterSpacing: "0.1em", color, padding: "0.15rem 0.4rem", borderRadius: "0.25rem", background: `${color}15`, border: `1px solid ${color}30` }}>
          {icon}
        </span>
        <span className="live-dot" style={{ backgroundColor: color }} />
      </div>
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color }}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
      <div className="stat-bg-glow" style={{ background: `radial-gradient(circle, ${color}, transparent)` }} />
    </motion.div>
  );
}
