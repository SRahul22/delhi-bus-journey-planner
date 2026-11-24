import os, subprocess, sys
PY = r"E:\hackathon-project\planner\.venv\Scripts\python.exe"
D = r"E:\hackathon-wt\planner\planner"
OUT = r"E:\hackathon-work\v2\planner"
n = sys.argv[1] if len(sys.argv) > 1 else "200"
base = dict(os.environ, MODEL_DIR=r"E:\hackathon-project\model", REALTIME="0")
runs = {
    "v1_flags_off": {},
    "all_flags_on": dict(V2_WAIT="pooled", V2_BAND="1", V3_PREDICT_DIR=OUT + r"\mlpredict",
                         V2_BAND_DIR=OUT + r"\mlpredict\v2", ADMIN_TOKEN="bench"),
}
for name, env in runs.items():
    e = dict(base, **env)
    print("==", name, flush=True)
    subprocess.run([PY, os.path.join(D, "bench_latency.py"), n, os.path.join(OUT, f"latency_{name}.json")], env=e, cwd=D)
