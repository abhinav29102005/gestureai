import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
// @ts-expect-error - parquetjs-lite lacks types
import { ParquetReader } from "parquetjs-lite";

const RESULTS_DIR = path.resolve(process.cwd(), "public/data/results");

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
  let raw: Frame[] = [];

  const parquetPath = path.join(RESULTS_DIR, `sub_${id}.parquet`);
  
  if (fs.existsSync(parquetPath)) {
    try {
      const parquet = require("parquetjs-lite");
      const reader = await parquet.ParquetReader.openFile(parquetPath);
      const cursor = reader.getCursor();
      let record = null;
      
      while (record = await cursor.next()) {
        try {
          // Parquetjs-lite represents lists as { list: [{element: v}, ...] }
          // and may return numbers as strings depending on the schema.
          const flattened: Frame = {
            activity: record.activity,
            confidence: parseFloat(record.confidence),
            attention: record.attention?.list?.map((i: any) => parseFloat(i.element)) || [],
            local_imp: record.local_imp?.list?.map((i: any) => parseFloat(i.element)) || [],
            raw_window: record.raw_window?.list?.map((i: any) => 
              i.element?.list?.map((j: any) => parseFloat(j.element)) || []
            ) || []
          };
          raw.push(flattened);
        } catch (err) {
          console.error("Error flattening record:", err);
        }
      }
      await reader.close();
    } catch (e) {
      console.error(`Error reading parquet for ${id}:`, e);
      return NextResponse.json({ 
        error: "Error reading data", 
        details: String(e),
        path: parquetPath 
      }, { status: 500 });
    }
  }

  if (raw.length === 0) {
    return NextResponse.json({ error: "Subject not found or no data" }, { status: 404 });
  }

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
        attention: Array.isArray(frame.attention) ? frame.attention.slice(0, 20) : [], // compact
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
