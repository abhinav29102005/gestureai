"use client";
import { useEffect, useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import StatCard from "@/components/StatCard";
import GlobalView from "@/components/GlobalView";
import SubjectView from "@/components/SubjectView";
import ImprovementsView from "@/components/ImprovementsView";
import { GlobalStats, SubjectData, formatNumber, getActivityColor } from "@/lib/types";

type Tab = "overview" | "subject" | "improvements";
const PILLS = [
  { label: "Val Accuracy", value: "94.2%", color: "#34d399" },
  { label: "Inference",    value: "<15ms", color: "#00f2fe" },
  { label: "Dataset",      value: "3.6M+", color: "#a78bfa" },
  { label: "Classes",      value: "18",    color: "#fbbf24" },
];

export default function Home() {
  const [subjects,     setSubjects]     = useState<string[]>([]);
  const [selSub,       setSelSub]       = useState("");
  const [globalStats,  setGlobalStats]  = useState<GlobalStats | null>(null);
  const [subjectData,  setSubjectData]  = useState<SubjectData | null>(null);
  const [activeTab,    setActiveTab]    = useState<Tab>("overview");
  const [loading,      setLoading]      = useState(true);
  const [subLoading,   setSubLoading]   = useState(false);
  const [gLoading,     setGLoading]     = useState(true);
  const [typed,        setTyped]        = useState("");
  const TITLE = "GesturAI Industrial Suite";

  /* typing animation */
  useEffect(() => {
    let i = 0;
    const t = setInterval(() => { setTyped(TITLE.slice(0, ++i)); if (i >= TITLE.length) clearInterval(t); }, 55);
    return () => clearInterval(t);
  }, []);

  /* fetch subjects */
  useEffect(() => {
    fetch("/api/subjects").then(r => r.json()).then(d => {
      const list = d.subjects ?? [];
      setSubjects(list);
      if (list.length) setSelSub(list[0]);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  /* fetch global */
  useEffect(() => {
    fetch("/api/global-stats").then(r => r.json()).then(d => {
      setGlobalStats(d); setGLoading(false);
    }).catch(() => setGLoading(false));
  }, []);

  /* fetch subject */
  const loadSub = useCallback((id: string) => {
    if (!id) return;
    setSubLoading(true); setSubjectData(null);
    fetch(`/api/subject/${id}`).then(r => r.json()).then(d => {
      setSubjectData(d); setSubLoading(false);
    }).catch(() => setSubLoading(false));
  }, []);

  useEffect(() => { if (selSub) loadSub(selSub); }, [selSub, loadSub]);

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      
      <main className="app-main">
        <TopBar typedTitle={typed} />
        
        <div className="scroll-content">
          <div className="wrapper">
            
            {/* Header / Subject Picker Row */}
            <div className="section" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div className="perf-pills">
                {PILLS.map(p => (
                  <div key={p.label} className="pill">
                    <span className="pill-value" style={{ color: p.color }}>{p.value}</span>
                    <span className="pill-label">{p.label}</span>
                  </div>
                ))}
              </div>

              <div className="subject-picker">
                <label htmlFor="sub-select">Subject Context</label>
                <select id="sub-select" value={selSub} disabled={loading}
                  onChange={e => { setSelSub(e.target.value); setActiveTab("subject"); }}>
                  {subjects.map(s => <option key={s} value={s}>Subject {s}</option>)}
                </select>
              </div>
            </div>

            {/* KPI CARDS */}
            <div className="section">
              {gLoading ? (
                <div className="grid-4">
                  {[...Array(4)].map((_, i) => <div key={i} className="shimmer" style={{ height: 130 }} />)}
                </div>
              ) : globalStats ? (
                <div className="grid-4">
                  <StatCard label="Subjects Analyzed"  value={String(globalStats.totalSubjects ?? 0)}     sub="WISDM v2.0 dataset"     glow="cyan"   delay={0}   />
                  <StatCard label="Total Windows"       value={formatNumber(globalStats.totalFrames)}      sub="Across all 51 subjects" glow="blue"   delay={80}  />
                  <StatCard label="Validation Accuracy" value="94.2%"                                      sub="On 18 activity classes" glow="green"  delay={160} />
                  <StatCard label="Inference Latency"   value="<15ms"                                      sub="CPU · ONNX optimized"   glow="purple" delay={240} />
                </div>
              ) : null}
            </div>

            {/* ACTIVITY LEGEND */}
            {globalStats && !gLoading && (
              <div className="section">
                <div className="legend-strip">
                  {(globalStats.activityBreakdown ?? []).map(a => (
                    <span key={a.name} className="badge"
                      style={{ color: getActivityColor(a.name), backgroundColor: `${getActivityColor(a.name)}16`, borderColor: `${getActivityColor(a.name)}40` }}>
                      <span className="badge-dot" style={{ background: getActivityColor(a.name) }} />
                      {a.name}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* TAB CONTENT */}
            <AnimatePresence mode="wait">
              {activeTab === "overview" && (
                <motion.div key="overview"
                  initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.25 }}>
                  {gLoading ? (
                    <div className="grid-2">
                      {[...Array(4)].map((_, i) => <div key={i} className="shimmer" style={{ height: 320 }} />)}
                    </div>
                  ) : globalStats ? (
                    <GlobalView stats={globalStats} />
                  ) : null}
                </motion.div>
              )}

              {activeTab === "subject" && (
                <motion.div key="subject"
                  initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.25 }}>
                  {subLoading ? (
                    <div className="grid-2">
                      {[...Array(6)].map((_, i) => <div key={i} className="shimmer" style={{ height: 200 }} />)}
                    </div>
                  ) : subjectData ? (
                    <SubjectView data={subjectData} />
                  ) : null}
                </motion.div>
              )}

              {activeTab === "improvements" && (
                <motion.div key="improvements"
                  initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.25 }}>
                  <ImprovementsView />
                </motion.div>
              )}
            </AnimatePresence>

          </div>
          
          <footer className="site-footer">
            <div className="wrapper">
              GesturAI Industrial Suite · Multi-Scale ResNet CNN-LSTM + Temporal Attention + SHAP XAI · WISDM v2.0 · 3,600,000+ records
            </div>
          </footer>
        </div>
      </main>
    </div>
  );
}
