"use client";
import { GlobalStats, getActivityColor, formatNumber } from "@/lib/types";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  PieChart, Pie, Legend,
} from "recharts";
import SensorImpBar from "./SensorImpBar";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ChartTip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="card" style={{ padding: "0.6rem 0.9rem", minWidth: 120, border: "1px solid #2a3f5f" }}>
      <div style={{ fontWeight: 700, marginBottom: 3, color: "#e2e8f0", fontSize: 12 }}>{label}</div>
      <div style={{ color: "#00f2fe", fontSize: 11 }}>{payload[0].value} windows</div>
    </div>
  );
};

export default function GlobalView({ stats }: { stats: GlobalStats }) {
  const top10 = [...stats.activityBreakdown].sort((a, b) => b.count - a.count).slice(0, 10);
  const radarData = stats.activityBreakdown.map(a => ({ subject: a.name, confidence: parseFloat((a.avgConf * 100).toFixed(1)) }));
  const pieData   = stats.activityBreakdown.map(a => ({ name: a.name, value: a.count }));
  const avgConf   = stats.activityBreakdown.reduce((s, a) => s + a.avgConf, 0) / stats.activityBreakdown.length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>

      {/* Row 1 */}
      <div className="grid-2">
        <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <div className="card-title">Activity Window Count — Top 10</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={top10} margin={{ top: 4, right: 16, bottom: 52, left: 0 }}>
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 11 }} angle={-40} textAnchor="end" interval={0} />
              <YAxis tick={{ fill: "#475569", fontSize: 11 }} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="count" radius={[5, 5, 0, 0]} maxBarSize={50}>
                {top10.map(a => <Cell key={a.name} fill={getActivityColor(a.name)} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div className="card-title">Average Confidence per Activity — Radar</div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} outerRadius="72%">
              <PolarGrid stroke="#1e2d45" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: "#475569", fontSize: 10 }} />
              <Radar dataKey="confidence" stroke="#00f2fe" fill="#00f2fe" fillOpacity={0.1} strokeWidth={2}
                dot={{ fill: "#00f2fe", r: 3 }} />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Row 2 */}
      <div className="grid-2">
        <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <div className="card-title">Global Activity Distribution</div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="48%" innerRadius={70} outerRadius={110}
                paddingAngle={2} dataKey="value" stroke="none">
                {pieData.map(a => <Cell key={a.name} fill={getActivityColor(a.name)} />)}
              </Pie>
              <Legend iconType="circle" iconSize={8}
                formatter={v => <span style={{ color: "#94a3b8", fontSize: 11 }}>{v}</span>} />
              <Tooltip formatter={v => [v, "windows"]}
                contentStyle={{ background: "#111827", border: "1px solid #1e2d45", borderRadius: 10, fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>

        <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
          style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
          <SensorImpBar values={stats.avgSensorImp} title="Global Sensor Importance (Avg XAI)" />
          <div className="insight">
            <strong style={{ color: "#00f2fe" }}>Key Insight: </strong>
            Accelerometer Resultant (Acc Res) is the #1 predictor at{" "}
            <strong style={{ color: "#e2e8f0" }}>{(stats.avgSensorImp[3] * 100).toFixed(1)}%</strong>.
            Gyro Z dominates rotational gestures like dribbling & kicking.
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.625rem" }}>
            {[
              { label: "Subjects",   v: stats.totalSubjects,            c: "#00f2fe" },
              { label: "Windows",    v: formatNumber(stats.totalFrames), c: "#4f8ef7" },
              { label: "Avg Conf",   v: `${(avgConf * 100).toFixed(1)}%`, c: "#34d399" },
            ].map(s => (
              <div key={s.label} style={{ textAlign: "center", padding: "0.75rem", borderRadius: "0.75rem", background: "#0d1421", border: "1px solid #1e2d45" }}>
                <div style={{ fontSize: "1.5rem", fontWeight: 900, color: s.c }}>{s.v}</div>
                <div style={{ fontSize: "0.65rem", color: "#475569", marginTop: 3, textTransform: "uppercase", letterSpacing: "0.08em" }}>{s.label}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Row 3 — Leaderboard */}
      <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
        <div className="card-title">Subject Leaderboard — Avg Confidence</div>
        <div className="leaderboard">
          {[...stats.subjectSummaries].sort((a, b) => b.avgConf - a.avgConf).map((s, i) => (
            <div key={s.id} className="lb-card">
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ fontSize: "0.65rem", color: "#475569" }}>#{i + 1}</span>
                <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "#e2e8f0" }}>{s.id}</span>
              </div>
              <div style={{ fontSize: "1.6rem", fontWeight: 900, color: s.avgConf > 0.7 ? "#34d399" : s.avgConf > 0.5 ? "#fbbf24" : "#ef4444" }}>
                {(s.avgConf * 100).toFixed(0)}%
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: getActivityColor(s.topActivity), display: "inline-block", flexShrink: 0 }} />
                <span style={{ fontSize: "0.65rem", color: "#64748b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.topActivity}</span>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
