@echo off
REM Start the Delhi bus journey planner on http://localhost:%PORT% (default 8000)
REM Env overrides: PYTHON, GTFS_DIR, MODEL_DIR, PORT
cd /d "%~dp0"
if "%PORT%"=="" set PORT=8000
if "%PYTHON%"=="" set PYTHON=python
if not exist .venv\Scripts\python.exe (
  %PYTHON% -m venv .venv || exit /b 1
  .venv\Scripts\python.exe -m pip install -q -r requirements.txt || exit /b 1
)
if not exist cache\network.pkl .venv\Scripts\python.exe preprocess.py || exit /b 1
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port %PORT%
