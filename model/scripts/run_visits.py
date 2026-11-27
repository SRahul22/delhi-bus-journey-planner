"""Run stop_visits.py for many days, N in parallel. Usage: python run_visits.py N DD,DD,..."""
import sys, subprocess, time, os
N = int(sys.argv[1]); days = sys.argv[2].split(",")
HERE = os.path.dirname(os.path.abspath(__file__))
LOGD = os.path.join(os.environ.get("WORK_DIR", r"E:\hackathon-work"), "v2", "data", "logs"); os.makedirs(LOGD, exist_ok=True)
running = []
while days or running:
    while days and len(running) < N:
        d = days.pop(0)
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "stop_visits.py"), d], stdout=open(os.path.join(LOGD, f"visits_{d}.log"), "w"), stderr=subprocess.STDOUT)
        running.append((d, p)); print(time.strftime("%H:%M:%S"), "start", d, flush=True)
    time.sleep(2)
    for d, p in list(running):
        if p.poll() is not None:
            running.remove((d, p)); print(time.strftime("%H:%M:%S"), "end", d, "rc", p.returncode, flush=True)
print("ALL DONE", flush=True)
