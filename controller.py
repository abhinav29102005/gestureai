import subprocess
import time
import sys
import os
import shutil

# --- INDUSTRIAL CONFIG ---
PYTHON_EXE = sys.executable
API_SCRIPT = "api.py"
DASH_SCRIPT = "dashboard.py"
TRAIN_SCRIPT = "main.py"
CACHE_FILE = "data_cache.joblib"
OUTPUT_DIR = "outputs"

processes = []

def start_api():
    print("🚀 Starting Production Sensor API...")
    p = subprocess.Popen([PYTHON_EXE, API_SCRIPT])
    processes.append(p)
    return p

def start_dashboard():
    print("📈 Starting Premium Industrial Dashboard...")
    # Use python -m streamlit to stay in the venv
    p = subprocess.Popen([PYTHON_EXE, "-m", "streamlit", "run", DASH_SCRIPT])
    processes.append(p)
    return p

def run_training(fresh=False):
    if fresh:
        print("🧹 Cleaning cache for a fresh production run...")
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE)
        if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    print("🧠 Starting Full-Scale Industrial Training Pipeline...")
    subprocess.run([PYTHON_EXE, TRAIN_SCRIPT], check=True)

def cleanup():
    print("\n🛑 Shutting down production environment...")
    for p in processes:
        p.terminate()
    print("✅ System offline.")

def main():
    if len(sys.argv) < 2:
        print("Industrial Controller Usage:")
        print("  python controller.py train        - Run pipeline (uses cache if exists)")
        print("  python controller.py train-fresh  - Clear cache and run full pipeline")
        print("  python controller.py dev          - Start API and Dashboard")
        print("  python controller.py all          - Full fresh run + deploy")
        return

    cmd = sys.argv[1].lower()

    try:
        if cmd == "train":
            run_training(fresh=False)
        elif cmd == "train-fresh":
            run_training(fresh=True)
        elif cmd == "dev":
            start_api()
            time.sleep(3) # Wait for scale
            start_dashboard()
            while True: time.sleep(1)
        elif cmd == "all":
            run_training(fresh=True)
            start_api()
            time.sleep(3)
            start_dashboard()
            while True: time.sleep(1)
        else:
            print(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"System Error: {e}")
        cleanup()

if __name__ == "__main__":
    main()
