#!/bin/bash
# Copy the v3 artefacts (small) from Windows E:\hackathon-work\v3 to model/v3 on the Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
O=(-o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o BatchMode=yes -o LogLevel=ERROR)
H=<user>@<windows-host>:E:/hackathon-work/v3
mkdir -p work
for f in v3_lookup.npz lgbm_model.json lgbm_model.txt calib.json results.json results.txt results_try1.txt importance.csv; do
  scp "${O[@]}" "$H/$f" . 2>/dev/null || echo "missing $f"
done
scp "${O[@]}" "$H/check_rides.parquet" "$H/test_preds.parquet" work/
mv importance.csv importance.txt 2>/dev/null || true   # *.csv is not versioned in model/; keep it as .txt
# self-test set for predict.py (json, so it is synced)
python3 - <<'EOF'
import pandas as pd, json
c = pd.read_parquet("work/check_rides.parquet").head(500)
c["depart_ist"] = pd.to_datetime(c.t0 + 19800, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
cols = ["route_id", "board_stop", "alight_stop", "depart_ist", "actual_s", "base_med", "base_mean", "pred_lgbm", "pred_calib"]
json.dump([{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in c[cols].to_dict("records")],
          open("check_rides.json", "w"))
print("check_rides.json", len(c))
EOF
ls -la
