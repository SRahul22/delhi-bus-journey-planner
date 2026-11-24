# Planner STATUS

- venv created (.venv: fastapi, uvicorn, pandas, numpy, scipy) - OK, no blocks
- preprocess.py done: cache/network.pkl (2574 routes, 17.8k footpath pairs) in 7s
- ttmodel.py (pluggable TravelTimeModel) written; router/app in progress
- 16:40 app.py + UI + run.sh working; /journey ~100-260ms (schedule-based)
- 16:41 evaluate.py (holdout + google pairs) + README done; model integration tested on synthetic CSVs; waiting for model/segment_times.csv
- 16:42 baseline (schedule) sanity results recorded; server stopped; polling model/ for segment_times.csv
- 16:46 INTEGRATED model v1 (day 15): 81% segs covered; added per-segment ratio fallback + weekday->weekend borrowing; sanity results changed (see final report). Waiting for full run / holdout_trips.csv
- 16:46 portable paths (paths.py), run.bat, requirements.txt; synced to E:\hackathon-project
- 17:15 model v1.5 integrated (auto-loaded at startup); tests/sanity.py added; waiting for holdout_trips.csv (~17:50)
- 17:27 realtime layer integrated (patch applied), UI live toggle, README/LOG updated
- 17:29 holdout eval v1.5: MdAPE 13.4%, 66% within 20% (schedule 32.5%/33%)
- 17:36 full model + blend(600s) integrated; ranking fixes; eval/realtime re-run; see docs/LOG.md
