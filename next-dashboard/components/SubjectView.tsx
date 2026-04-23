"use client";
import { SubjectData, getActivityColor } from "@/lib/types";
import { motion } from "framer-motion";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, CartesianGrid,
} from "recharts";
import ActivityBadge from "./ActivityBadge";
import ConfidenceGauge from "./ConfidenceGauge";
import AttentionHeatmap from "./AttentionHeatmap";
import SensorImpBar from "./SensorImpBar";

export default function SubjectView({ data }: { data: SubjectData }) {
  const sorted     = [...data.activityStats].sort((a, b) => b.count - a.count);
  const top        = sorted[0];
  const avgConf    = data.activityStats.reduce((s, a) => s + a.avgConf * a.count, 0) / data.totalFrames;
  const bestAct    = data.activityStats.reduce((p, c) => c.avgConf > p.avgConf ? c : p);
  const worstAct   = data.activityStats.reduce((p, c) => c.avgConf < p.avgConf ? c : p);
  const lastFrame  = data.timeline[data.timeline.length - 1];

  const confLine   = data.timeline.map(t => ({ i: t.index, v: parseFloat((t.confidence * 100).toFixed(1)) }));
  const scatter    = data.timeline.map(t => ({ x: t.index, y: parseFloat((t.confidence * 100).toFixed(1)) }));

  const gap = { display: "flex", flexDirection: "column" as const, gap: "1.5rem" };

  return (
    <div style={gap}>

      {/* Row 1 — 4 kpi mini */}
      <div className="grid-4">
        {[
          { label: "Total Windows",   value: data.totalFrames,               color: "#00f2fe" },
          { label: "Avg Confidence",  value: `${(avgConf*100).toFixed(1)}%`, color: "#34d399" },
          { label: "Top Activity",    value: top?.name ?? "—",               color: getActivityColor(top?.name ?? "") },
          { label: "Activities Found",value: data.activityStats.length,      color: "#a78bfa" },
        ].map((s, i) => (
          <motion.div key={s.label} className="card" style={{ textAlign: "center" }}
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}>
            <div style={{ fontSize: "2rem", fontWeight: 900, color: s.color }}>{s.value}</div>
            <div style={{ fontSize: "0.65rem", textTransform: "uppercase", letterSpacing: "0.1em", color: "#475569", marginTop: 6 }}>{s.label}</div>
          </motion.div>
        ))}
      </div>

      {/* Row 2 — Confidence timeline + gauge */}
      <div className="grid-3-1">
        <motion.div className="card" initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          <div className="card-title">Confidence Timeline</div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={confLine} margin={{ top: 4, right: 16, bottom: 20, left: 0 }}>
              <defs>
                <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#00f2fe" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00f2fe" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d45" />
              <XAxis dataKey="i" tick={{ fill: "#475569", fontSize: 11 }} label={{ value: "Window Index", position: "insideBottom", offset: -8, fill: "#475569", fontSize: 10 }} />
              <YAxis tick={{ fill: "#475569", fontSize: 11 }} domain={[0, 100]} unit="%" />
              <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1e2d45", borderRadius: 10, fontSize: 12 }} formatter={v => [`${v}%`, "Confidence"]} />
              <Area type="monotone" dataKey="v" stroke="#00f2fe" strokeWidth={2} fill="url(#cg)" dot={false} activeDot={{ r: 5, fill: "#00f2fe" }} />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div className="card" initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.15 }}
          style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "1rem" }}>
          <div className="card-title" style={{ marginBottom: 0 }}>Last Window</div>
          {lastFrame && <>
            <ConfidenceGauge value={lastFrame.confidence} size={160} />
            <ActivityBadge activity={lastFrame.activity} confidence={lastFrame.confidence} size="lg" />
          </>}
        </motion.div>
      </div>

      {/* Row 3 — Activity bars + Sensor imp */}
      <div className="grid-2">
        <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <div className="card-title">Activity Breakdown</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem", maxHeight: "18rem", overflowY: "auto", paddingRight: 4 }}>
            {sorted.map(act => {
              const color = getActivityColor(act.name);
              const pct   = (act.count / data.totalFrames) * 100;
              return (
                <div key={act.name}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: 5 }}>
                    <span style={{ fontWeight: 600, color }}>{act.name}</span>
                    <span style={{ color: "#64748b" }}>{act.count} · {(act.avgConf * 100).toFixed(0)}%</span>
                  </div>
                  <div className="prog-track">
                    <motion.div className="prog-fill"
                      initial={{ width: 0 }} animate={{ width: `${pct}%` }}
                      transition={{ duration: 0.9, ease: "easeOut" }}
                      style={{ background: `linear-gradient(90deg, ${color}55, ${color})` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>

        <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
          <SensorImpBar values={top?.avgImp ?? new Array(8).fill(0.125)} title={`Sensor Attribution — ${top?.name ?? ""}`} />
        </motion.div>
      </div>

      {/* Row 4 — Attention heatmap */}
      {lastFrame && (
        <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <AttentionHeatmap values={lastFrame.attention} />
        </motion.div>
      )}

      {/* Row 5 — Scatter */}
      <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
        <div className="card-title">Confidence Distribution — All Windows</div>
        <ResponsiveContainer width="100%" height={230}>
          <ScatterChart margin={{ top: 4, right: 16, bottom: 16, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2d45" />
            <XAxis dataKey="x" name="Window"     tick={{ fill: "#475569", fontSize: 11 }} />
            <YAxis dataKey="y" name="Confidence" tick={{ fill: "#475569", fontSize: 11 }} unit="%" domain={[0, 100]} />
            <ZAxis range={[28, 28]} />
            <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1e2d45", borderRadius: 10, fontSize: 12 }}
              formatter={(v, n) => [`${v}${n === "Confidence" ? "%" : ""}`, n]} />
            <Scatter data={scatter} fill="#4f8ef7" opacity={0.7} />
          </ScatterChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Row 6 — Best / Worst */}
      <div className="grid-2">
        {[
        { act: bestAct,  label: "Highest Confidence", bc: "rgba(52,211,153,0.3)",  bs: "0 0 24px rgba(52,211,153,0.12)", c: "#34d399" },
          { act: worstAct, label: "Lowest Confidence",  bc: "rgba(239,68,68,0.3)",   bs: "0 0 24px rgba(239,68,68,0.12)", c: "#ef4444" },
        ].map(r => (
          <motion.div key={r.label} className="card" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
            style={{ borderColor: r.bc, boxShadow: r.bs }}>
            <div style={{ fontSize: "0.65rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em", color: r.c, marginBottom: "0.75rem" }}>{r.label}</div>
            <div style={{ display: "flex", alignItems: "center", gap: "1.25rem" }}>
              <ActivityBadge activity={r.act.name} size="lg" />
              <div style={{ fontSize: "3.5rem", fontWeight: 900, color: r.c }}>{(r.act.avgConf * 100).toFixed(1)}%</div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
