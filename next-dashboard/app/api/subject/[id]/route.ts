import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const RESULTS_DIR = path.resolve(process.cwd(), "data/results");

type Frame = {
  activity: string;
  confidence: number;
  attention: number[];
  local_imp: number[];
  raw_window: number[][];
};

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const filePath = path.join(RESULTS_DIR, `sub_${id}.json`);

  if (!fs.existsSync(filePath)) {
    return NextResponse.json({ error: "Subject not found" }, { status: 404 });
  }

  const raw: Frame[] = JSON.parse(fs.readFileSync(filePath, "utf-8"));

  // Compute per-activity stats
  const activityMap: Record<string, { count: number; totalConf: number; totalImp: number[] }> = {};
  const timeline: { index: number; activity: string; confidence: number; attention: number[]; local_imp: number[] }[] = [];

  raw.forEach((frame, i) => {
    if (!activityMap[frame.activity]) {
      activityMap[frame.activity] = { count: 0, totalConf: 0, totalImp: new Array(8).fill(0) };
    }
    activityMap[frame.activity].count++;
    activityMap[frame.activity].totalConf += frame.confidence;
    frame.local_imp.forEach((v, j) => { activityMap[frame.activity].totalImp[j] += v; });

    // Downsample timeline to max 200 frames for performance
    if (i % Math.max(1, Math.floor(raw.length / 200)) === 0) {
      timeline.push({
        index: i,
        activity: frame.activity,
        confidence: frame.confidence,
        attention: frame.attention.slice(0, 20), // compact
        local_imp: frame.local_imp,
      });
    }
  });

  const activityStats = Object.entries(activityMap).map(([name, d]) => ({
    name,
    count: d.count,
    avgConf: d.totalConf / d.count,
    avgImp: d.totalImp.map((v) => v / d.count),
  }));

  return NextResponse.json({
    subjectId: id,
    totalFrames: raw.length,
    activityStats,
    timeline,
  });
}
