# model/v3 status (ML refinement of ride times) — handoff ~18:05

Stage: DONE. Experiments, packaging and write-up are complete; nothing is running.

Results (600k held-out rides, 25-31 Oct, median APE / within ±20%): sum of medians 12.69% / 67.8%;
median→mean blend 12.45% / 68.3%; bucket calibration 12.43% / 68.4%; LightGBM L1 (shipped) 12.32% / 68.9%;
XGBoost GPU 12.32% / 68.8%. The ML gain over calibration is ~0.1 pt, which is not meaningful.
Recommendation: keep the planner's blend; v3 predict.py is optional (INTEGRATION.md, not applied).

Artefacts in model/v3: predict.py (self-test passes: `python predict.py`), v3_lookup.npz, lgbm_model.json/.txt,
calib.json, check_rides.json, results*.txt, importance.txt. Work files and parquet data: E:\hackathon-work\v3.

Next (optional): re-run train.py once to get the planner_blend600 row in results.txt (~4 min, detached via
launch.py); decide on integration; consider combining with the realtime ratios (README "Next steps").
