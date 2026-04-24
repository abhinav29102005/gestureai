import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const RESULTS_DIR = path.resolve(process.cwd(), "public/data/results");

export async function GET() {
  try {
    const manifestPath = path.join(RESULTS_DIR, "manifest.json");
    if (fs.existsSync(manifestPath)) {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
      return NextResponse.json(manifest);
    }
    
    // Fallback if manifest not found
    const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith(".parquet") && f.startsWith("sub_"));
    const subjects = files.map((f) => f.replace("sub_", "").replace(".parquet", "")).sort();
    return NextResponse.json({ subjects });
  } catch (error) {
    console.error("Error listing subjects:", error);
    return NextResponse.json({ subjects: [], error: "Could not read results directory" }, { status: 500 });
  }
}
