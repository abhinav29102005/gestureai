import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const RESULTS_DIR = path.resolve(process.cwd(), "data/results");

type Frame = { activity: string; confidence: number; local_imp: number[] };

export async function GET() {
  try {
    const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith(".json"));

    const globalActivity: Record<string, { count: number; totalConf: number }> = {};
    const globalImp = new Array(8).fill(0);
    let impCount = 0;
    let totalFrames = 0;
    const subjectSummaries: { id: string; totalFrames: number; topActivity: string; avgConf: number }[] = [];

    for (const file of files) {
      const id = file.replace("sub_", "").replace(".json", "");
      const raw: Frame[] = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, file), "utf-8"));
      totalFrames += raw.length;

      const subConf: number[] = [];
      const subActivityCount: Record<string, number> = {};

      raw.forEach((frame) => {
        if (!globalActivity[frame.activity]) globalActivity[frame.activity] = { count: 0, totalConf: 0 };
        globalActivity[frame.activity].count++;
        globalActivity[frame.activity].totalConf += frame.confidence;
        subConf.push(frame.confidence);
        subActivityCount[frame.activity] = (subActivityCount[frame.activity] || 0) + 1;
        frame.local_imp.forEach((v, i) => { globalImp[i] += v; });
        impCount++;
      });

      const topActivity = Object.entries(subActivityCount).sort((a, b) => b[1] - a[1])[0]?.[0] || "unknown";
      const avgConf = subConf.reduce((a, b) => a + b, 0) / subConf.length;
      subjectSummaries.push({ id, totalFrames: raw.length, topActivity, avgConf });
    }

    const activityBreakdown = Object.entries(globalActivity).map(([name, d]) => ({
      name,
      count: d.count,
      avgConf: d.totalConf / d.count,
    })).sort((a, b) => b.count - a.count);

    const avgSensorImp = globalImp.map((v) => v / impCount);

    return NextResponse.json({
      totalSubjects: files.length,
      totalFrames,
      activityBreakdown,
      avgSensorImp,
      subjectSummaries,
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
