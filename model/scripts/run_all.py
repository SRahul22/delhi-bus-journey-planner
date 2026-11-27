"""Run process_day.py for many days in parallel. Usage: python run_all.py NPROC DD,DD,..."""
import subprocess, sys, os, time
from concurrent.futures import ThreadPoolExecutor
W = os.environ.get("WORK_DIR", r"E:\hackathon-work")
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(W, "pass"), exist_ok=True)
# process_day.py next to this script (the v1 run used a copy in WORK_DIR; same code)
PD = os.path.join(HERE, "process_day.py") if os.path.exists(os.path.join(HERE, "process_day.py")) else os.path.join(W, "process_day.py")
n = int(sys.argv[1]); days = sys.argv[2].split(",")
def run(d):
    fn = os.path.join(W, "pass", f"pass_{d}.npz")
    if os.path.exists(fn) and os.path.getsize(fn) > 1000:   # 0-byte files are "claimed" placeholders
        return d, "exists"
    with open(os.path.join(W, f"log_{d}.txt"), "w") as f:
        r = subprocess.run([sys.executable, PD, d], stdout=f, stderr=subprocess.STDOUT, cwd=W)
    return d, r.returncode
t = time.time()
with ThreadPoolExecutor(n) as ex:
    for d, rc in ex.map(run, days):
        print(f"{time.time()-t:7.0f}s day {d} rc={rc}", flush=True)
print("ALL DONE", flush=True)
