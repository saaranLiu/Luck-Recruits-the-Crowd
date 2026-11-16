"""Paths for the current estimates. Analysis files are in data/."""
import os, time

PROJ = os.path.dirname(os.path.abspath(__file__))
WORK = f"{PROJ}/data"
RES = f"{PROJ}/results"
T_END = 1777374040            # last block time in the tape (2026-04-28 11:00 UTC)
DAY = 86400

_T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s]", *a, flush=True)
