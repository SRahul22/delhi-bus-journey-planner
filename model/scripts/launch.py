"""Detached launcher: python launch.py <logfile> <args...>"""
import os, subprocess, sys
CWD = os.environ.get("WORK_DIR", r"E:\hackathon-work")
flags = 0x00000008 | 0x00000200 | 0x01000000  # DETACHED | NEW_GROUP | BREAKAWAY_FROM_JOB
log = open(sys.argv[1], "w")
try:
    p = subprocess.Popen([sys.executable] + sys.argv[2:], stdout=log, stderr=subprocess.STDOUT, creationflags=flags, cwd=CWD)
except OSError as e:
    print("breakaway failed", e); p = subprocess.Popen([sys.executable] + sys.argv[2:], stdout=log, stderr=subprocess.STDOUT, creationflags=0x8 | 0x200, cwd=CWD)
print("pid", p.pid)
