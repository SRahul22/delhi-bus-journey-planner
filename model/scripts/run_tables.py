"""v2 T0.4 (data): segment/headway tables with holiday day type. 1-18 (validation-time selection) then 1-24 (serving)."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, HOLIDAYS=os.environ.get("HOLIDAYS", "02,20"))
for name, last in (("tables_1_18", 18), ("tables_1_24", 24)):
    days = ",".join(f"{d:02d}" for d in range(1, last + 1))
    print(time.strftime("%H:%M:%S"), "start", name, flush=True)
    rc = subprocess.call([sys.executable, os.path.join(HERE, "aggregate.py"), days, "-", os.path.join(r"E:\hackathon-work\v2\data", name)], env=env)
    print(time.strftime("%H:%M:%S"), "end", name, "rc", rc, flush=True)
