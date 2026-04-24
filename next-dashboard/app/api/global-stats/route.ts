import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const RESULTS_DIR = path.resolve(process.cwd(), "public/data/results");

export async function GET() {
  try {
    const statsPath = path.join(RESULTS_DIR, "global_stats.json");
    if (fs.existsSync(statsPath)) {
      const stats = JSON.parse(fs.readFileSync(statsPath, "utf-8"));
      return NextResponse.json(stats);
    }
    
    // Fallback/Empty state if not pre-calculated
    return NextResponse.json({
      totalSubjects: 0,
      totalFrames: 0,
      activityBreakdown: [],
      avgSensorImp: new Array(8).fill(0),
      subjectSummaries: []
    });
  } catch (e) {
    console.error("Global stats error:", e);
    return NextResponse.json({ error: "Failed to load global statistics" }, { status: 500 });
  }
}
