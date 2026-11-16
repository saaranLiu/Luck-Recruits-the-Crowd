"""Paths for this reproduction package.

Analysis-ready files are in ``data/``. Rebuilding ``data/exante.parquet`` or
``data/quality_r2.parquet`` from the Polymarket fill tape requires
``POLYMARKET_ROOT`` (a directory that contains ``database/daily_aligned``,
``database/OrderFilled``, and ``database/CTF``) and ``MARKETS_PARQUET`` (the
full market table). ``ANALYSIS_THREADS`` and ``ANALYSIS_MEMORY`` override the
DuckDB settings used by that rebuild.
"""
import duckdb, os, time

PROJ = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("POLYMARKET_ROOT", "")
RAW_TAPE = bool(ROOT)
WORK = f"{PROJ}/data"
RES = f"{PROJ}/results"
FIGS = f"{PROJ}/figures"
T_END = 1777374040            # last block timestamp in the tape (2026-04-28 11:00 UTC)
DAY = 86400
GROUPS = ('UpDown', 'SportsML', 'YesNo')

if RAW_TAPE:
    DA = os.environ.get("DA_GLOB", f"{ROOT}/database/daily_aligned/*.parquet")
    CTF = f"{ROOT}/database/CTF"
    MARKETS = os.environ.get("MARKETS_PARQUET", "")
    if not MARKETS:
        raise SystemExit("Set MARKETS_PARQUET to the full market table before reading the fill tape.")
else:
    DA = ""
    CTF = ""
    # Sports rows only (condition id, open time). Enough for the schedule audit, not for a tape rebuild.
    MARKETS = os.environ.get("MARKETS_PARQUET", f"{WORK}/markets_sports.parquet")

_T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s]", *a, flush=True)


def connect(threads=None, mem=None):
    if threads is None:
        threads = int(os.environ.get("ANALYSIS_THREADS", "8"))
    if mem is None:
        mem = os.environ.get("ANALYSIS_MEMORY", "32GB")
    tmp = f"{WORK}/duckdb_tmp"
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"SET threads={int(threads)}; SET memory_limit='{mem}'; SET preserve_insertion_order=false; "
        f"SET temp_directory='{tmp}'"
    )
    return con
