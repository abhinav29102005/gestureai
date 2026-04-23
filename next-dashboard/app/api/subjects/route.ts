import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const RESULTS_DIR = path.resolve(process.cwd(), "data/results");

export async function GET() {
  try {
    const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith(".json"));
    const subjects = files.map((f) => f.replace("sub_", "").replace(".json", "")).sort();
    return NextResponse.json({ subjects });
  } catch {
    return NextResponse.json({ subjects: [], error: "Could not read results directory" }, { status: 500 });
  }
}
