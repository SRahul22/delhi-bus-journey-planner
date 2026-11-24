"""Portable data locations. Override with env vars GTFS_DIR / MODEL_DIR."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def gtfs_dir():
    env = os.environ.get("GTFS_DIR")
    if env:
        return env
    for cand in (os.path.join(REPO, "data", "gtfs"), "E:/hackathon-data/gtfs"):
        if os.path.exists(os.path.join(cand, "stops.txt")):
            return cand
    return os.path.join(REPO, "data", "gtfs")


def model_dir():
    return os.environ.get("MODEL_DIR", os.path.join(REPO, "model"))
