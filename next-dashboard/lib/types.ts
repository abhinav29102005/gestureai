export type ActivityStat = {
  name: string;
  count: number;
  avgConf: number;
  avgImp?: number[];
};

export type TimelineEntry = {
  index: number;
  activity: string;
  confidence: number;
  attention: number[];
  local_imp: number[];
};

export type SubjectData = {
  subjectId: string;
  totalFrames: number;
  activityStats: ActivityStat[];
  timeline: TimelineEntry[];
};

export type SubjectSummary = {
  id: string;
  totalFrames: number;
  topActivity: string;
  avgConf: number;
};

export type GlobalStats = {
  totalSubjects: number;
  totalFrames: number;
  activityBreakdown: { name: string; count: number; avgConf: number }[];
  avgSensorImp: number[];
  subjectSummaries: SubjectSummary[];
};

export const SENSOR_LABELS = ["Acc X", "Acc Y", "Acc Z", "Acc Res", "Gyro X", "Gyro Y", "Gyro Z", "Gyro Res"];

export const ACTIVITY_COLORS: Record<string, string> = {
  walking: "#34d399",
  jogging: "#f472b6",
  stairs: "#60a5fa",
  sitting: "#fbbf24",
  standing: "#a78bfa",
  typing: "#00f2fe",
  writing: "#fb923c",
  pasta: "#f59e0b",
  sandwich: "#84cc16",
  chips: "#ec4899",
  folding: "#8b5cf6",
  catch: "#14b8a6",
  kicking: "#ef4444",
  dribbling: "#f97316",
  drinking: "#06b6d4",
  soup: "#d97706",
  clapping: "#10b981",
  teeth: "#6366f1",
};

export function getActivityColor(activity: string): string {
  return ACTIVITY_COLORS[activity.toLowerCase()] ?? "#94a3b8";
}

export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || isNaN(n)) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toString();
}
