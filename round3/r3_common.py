"""Shared paths for round-3 checks.  Round-3 outputs go to round3/; round-2 inputs are read, never overwritten."""
import os, sys
R3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(R3), 'round2'))
from r2_common import *          # noqa: E402,F401
R3RES, R3NOTE, R3WORK = (f'{R3}/{d}' for d in ('results', 'notes', 'work'))
SCHED = f'{WORK}/schedules'
