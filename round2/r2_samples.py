"""Round-2 samples.  Up/down: A, B, C as in round 1.  Sports: S0 (round-1 A, order >= 6 h before resolution) and S1
(order before the independent scheduled start, NBA/NHL/NFL), each with its B and C restrictions.

Day variables: oday = UTC day of the first order; rday = UTC day of the first market's resolution (the time origin of
every outcome).  Period: early / late halves of each domain's data (fixed in round 1).
"""
from r2_common import *
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(R2))
from prep_ex import load_ex, samples_ex, ols2, spec as spec1, SPEC_EX  # noqa: F401

SPLIT = {'UD': pd.Timestamp('2026-01-01').timestamp(), 'SP': pd.Timestamp('2025-07-01').timestamp()}


def load_r2():
    d = load_ex()
    t = pd.read_parquet(f'{WORK}/sp_timing.parquet')[['w', 'ind_start', 'S0', 'S1', 'S2', 'S3']]
    d = d.merge(t, on='w', how='left')
    for c in ['S0', 'S1', 'S2', 'S3']:
        d[c] = d[c].fillna(False).astype(bool)
    d['oday'] = (d.ts0 // DAY).astype(int)
    d['rday'] = (d.res_ts // DAY).astype(int)
    d['rweek'] = (d.res_ts // (7 * DAY)).astype(int)
    d['late'] = np.where(d.grp == 'UpDown', d.ts0 >= SPLIT['UD'], d.ts0 >= SPLIT['SP']).astype(float)
    d['cl_game'] = np.where(d.grp == 'UpDown', d.cid, d.event)
    return d


def samples_r2(d):
    S = samples_ex(d)
    held = d.held == 1
    C = held & (d.n_gap == 0)
    out = {'UD_A': S['UD_A'], 'UD_B': S['UD_B'], 'UD_C': S['UD_C'],
           'SP_S0': S['SP_A'], 'SP_S0B': S['SP_B'], 'SP_S0C': S['SP_C']}
    s1 = d.S1 & (d.grp == 'SportsML')
    out.update({'SP_S1': d[s1], 'SP_S1B': d[s1 & held], 'SP_S1C': d[s1 & C]})
    return out


def spec(k):
    return SPEC_EX[k[:2]]


MAIN = ['UD_A', 'SP_S1', 'SP_S0']
