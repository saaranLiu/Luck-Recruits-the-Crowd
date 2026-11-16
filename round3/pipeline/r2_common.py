"""Shared paths and helpers for the round-3 rerun of the round-2 pipeline.  Outputs go to round3/pipeline/."""
import os, sys, json, hashlib, time
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJ)
from common import *          # noqa: E402,F401  (ROOT, DA, MARKETS, WORK, RES, T_END, DAY, connect, log)

R2 = os.path.dirname(os.path.abspath(__file__))
R2RES, R2AUD, R2FIG, R2NOTE, R2WORK = (f'{R2}/{d}' for d in ('results', 'audit', 'figures', 'notes', 'work'))
OF = f'{ROOT}/database/OrderFilled'
RELAYERS = ('0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e', '0xc5d563a36ae78145c45a50134d48a1215220f80a')
SEED = 20261010


def ledger_rows(analysis_id, df, **meta):
    """Attach ledger metadata to a result frame (one row per estimate)."""
    df = df.copy()
    df.insert(0, 'analysis_id', [f'{analysis_id}:{i}' for i in range(len(df))])
    for k, v in meta.items():
        df[k] = v
    return df
