"""Current samples. Crypto A: first order before the price window opens. Sports S1: first order before the scheduled start."""
import numpy as np
import pandas as pd
from common import DAY, WORK
from prep_ex import load_ex, samples_ex

SPLIT = {'UD': pd.Timestamp('2026-01-01').timestamp(), 'SP': pd.Timestamp('2025-07-01').timestamp()}
SPEC = {
    'UD': dict(ctrl=('lstake', 'lsched'), cat=('pbin', 'cell', 'hour')),
    'SP': dict(ctrl=('lstake',), cat=('pbin', 'month', 'hour')),
}


def load_sample():
    d = load_ex()
    t = pd.read_parquet(f'{WORK}/sp_timing.parquet')[['w', 'ind_start', 'S1']]
    d = d.merge(t, on='w', how='left')
    d['S1'] = d['S1'].astype('boolean').fillna(False).astype(bool)
    d['lsched'] = np.log1p(d.tend - d.ts0)
    d['lead_s'] = np.where(d.S1, d.ind_start - d.ts0, d.lead)
    d['rday'] = (d.res_ts // DAY).astype(int)
    d['late'] = np.where(d.grp == 'UpDown', d.ts0 >= SPLIT['UD'], d.ts0 >= SPLIT['SP']).astype(float)
    return d


def samples(d):
    S = samples_ex(d)
    s1 = d.S1 & (d.grp == 'SportsML')
    return {'UD_A': S['UD_A'], 'SP_S1': d[s1]}


def spec(k):
    return SPEC[k[:2]]
