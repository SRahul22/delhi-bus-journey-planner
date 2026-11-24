# Windows: check_rides.parquet (from train.py) -> check_rides.json for predict.py's self-test
import pandas as pd, json
c = pd.read_parquet(r"E:\hackathon-work\v3\check_rides.parquet").head(500)
c["depart_ist"] = pd.to_datetime(c.t0 + 19800, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
cols = ["route_id", "board_stop", "alight_stop", "depart_ist", "actual_s", "base_med", "base_mean", "pred_lgbm", "pred_calib"]
json.dump([{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in c[cols].to_dict("records")],
          open(r"E:\hackathon-work\v3\check_rides.json", "w"))
print("ok")
