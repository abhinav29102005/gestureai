import os
import json
import math

DATA_DIR = "public/data/results"
MAX_FILE_SIZE = 24 * 1024 * 1024 # 24MB

def process_data():
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found.")
        return

    # Find all unique subject IDs
    subject_ids = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith(".json") and f.startswith("sub_"):
            sid = f.split("_p")[0].replace("sub_", "").replace(".json", "")
            subject_ids.add(sid)
    
    print(f"Found {len(subject_ids)} subjects. Calculating global stats...")

    global_activity = {}
    global_imp = [0.0] * 8
    imp_count = 0
    total_frames = 0
    subject_summaries = []

    for sid in sorted(list(subject_ids)):
        print(f"  Processing Subject {sid}...")
        sub_items = []
        
        # Try to load single file or chunks
        single_file = os.path.join(DATA_DIR, f"sub_{sid}.json")
        if os.path.exists(single_file):
            with open(single_file, 'r') as f:
                sub_items = json.load(f)
        else:
            chunk_idx = 0
            while True:
                chunk_path = os.path.join(DATA_DIR, f"sub_{sid}_p{chunk_idx}.json")
                if not os.path.exists(chunk_path):
                    break
                with open(chunk_path, 'r') as f:
                    sub_items.extend(json.load(f))
                chunk_idx += 1
        
        if not sub_items:
            print(f"    No data found for subject {sid}")
            continue

        total_frames += len(sub_items)
        sub_activity_count = {}
        sub_conf_total = 0.0
        
        for frame in sub_items:
            act = frame.get("activity", "unknown")
            conf = frame.get("confidence", 0.0)
            
            if act not in global_activity:
                global_activity[act] = {"count": 0, "totalConf": 0.0}
            global_activity[act]["count"] += 1
            global_activity[act]["totalConf"] += conf
            
            sub_activity_count[act] = sub_activity_count.get(act, 0) + 1
            sub_conf_total += conf
            
            if "local_imp" in frame:
                for i, v in enumerate(frame["local_imp"]):
                    if i < 8:
                        global_imp[i] += v
                imp_count += 1
        
        top_activity = max(sub_activity_count.items(), key=lambda x: x[1])[0] if sub_activity_count else "unknown"
        avg_conf = sub_conf_total / len(sub_items) if sub_items else 0
        
        subject_summaries.append({
            "id": sid,
            "totalFrames": len(sub_items),
            "topActivity": top_activity,
            "avgConf": avg_conf
        })

    # Finalize Global Stats
    activity_breakdown = []
    for name, d in global_activity.items():
        activity_breakdown.append({
            "name": name,
            "count": d["count"],
            "avgConf": d["totalConf"] / d["count"] if d["count"] > 0 else 0
        })
    activity_breakdown.sort(key=lambda x: x["count"], reverse=True)
    
    avg_sensor_imp = [v / imp_count if imp_count > 0 else 0 for v in global_imp]
    
    global_stats = {
        "totalSubjects": len(subject_ids),
        "totalFrames": total_frames,
        "activityBreakdown": activity_breakdown,
        "avgSensorImp": avg_sensor_imp,
        "subjectSummaries": sorted(subject_summaries, key=lambda x: x["id"])
    }
    
    with open(os.path.join(DATA_DIR, "global_stats.json"), 'w') as f:
        json.dump(global_stats, f)
    print(f"Generated global_stats.json")

    with open(os.path.join(DATA_DIR, "manifest.json"), 'w') as f:
        json.dump({"subjects": sorted(list(subject_ids))}, f)
    print(f"Generated manifest.json")

if __name__ == "__main__":
    process_data()
