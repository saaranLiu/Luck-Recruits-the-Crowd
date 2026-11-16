"""Round-3 rerun of the up/down results that round 2 took from round-1 scripts and that used the ex-post timing
control (ldelay): rich pre-bet controls (15_inference.py, part 2), in-sample associations of pre-bet features with
the win (14_random.py), and the regression band of the close-call figure (17_rd.py, 'regression, same rows').
Only the timing control changes: lsched = log1p(scheduled window end - order).  Rows, features, and clustering are
those of the round-1 scripts.
"""
from r2_common import *
import numpy as np, pandas as pd
from r2_samples import load_r2, samples_r2, spec, ols2
from prep_ex import H_S

SP_OLD = dict(ctrl=('lstake', 'ldelay'), cat=('pbin', 'cell', 'hour'))
RICH = ('r15m', 'r1h', 'r4h', 'r24h', 'vol24', 'prev1', 'prev4', 'lfills0', 'llead')
out = []

# ---- rich pre-bet controls and in-sample associations (round-1 feature files) ----------------------------------------
for k in ['UD_A', 'UD_C']:
    f = pd.read_parquet(f'{WORK}/{k}_features.parquet')
    f['lsched'] = np.log1p(f.tend - f.ts0)
    for c in ['r15m', 'r1h', 'r4h', 'r24h', 'vol24', 'prev1', 'prev4']:
        f[c] = f[c].fillna(f[c].median())
    f['asset_hz'] = f.asset + '_' + f.hz
    for lab, sp in [('M1 scheduled end (round 3)', spec(k)), ('M0 actual resolution (round 2)', SP_OLD)]:
        for y in ['ret7d', 'ret30d']:
            r = ols2(f, y, **sp)
            out.append(dict(part='rich controls', sample=k, timing=lab, spec='main (same rows)', **r))
            r = ols2(f, y, ctrl=sp['ctrl'] + RICH, cat=sp['cat'] + ('asset_hz', 'dow'))
            out.append(dict(part='rich controls', sample=k, timing=lab, spec='rich pre-bet controls', **r))
    raw = pd.read_parquet(f'{WORK}/{k}_features.parquet')
    for c in ['r15m', 'r1h', 'r4h', 'r24h', 'prev1', 'prev4', 'vol24']:
        vv = f.assign(fz=(raw[c] - raw[c].mean()) / raw[c].std())
        for lab, sp in [('M1 scheduled end (round 3)', spec(k)), ('M0 actual resolution (round 2)', SP_OLD)]:
            r = ols2(vv, 'win', x='fz', ctrl=sp['ctrl'], cat=sp['cat'])
            if r: out.append(dict(part='insample association', sample=k, timing=lab, spec=f'insample_{c}', **r))
    log(k, 'rich and in-sample done')

# ---- regression band of the close-call figure ---------------------------------------------------------------------------
d = load_r2()
S = samples_r2(d)
k15 = pd.read_parquet(f'{WORK}/klines/binance_15m.parquet')
k15['asset'] = k15.sym.str.replace('USDT', '').str.lower()
k15 = k15.sort_values(['asset', 't'])
k15['lr'] = k15.groupby('asset').close.transform(lambda s: np.log(s).diff())
k15['vol24'] = k15.groupby('asset').lr.transform(lambda s: s.rolling(96).std().shift(1))
OP = k15.set_index(['asset', 't']).open; CLS = k15.set_index(['asset', 't']).close; VOL = k15.set_index(['asset', 't']).vol24


def running(v):
    v = v[v.hz.isin(['15m', '4h']) & v.asset.isin(['btc', 'eth', 'sol', 'xrp'])].copy()
    hs = v.hz.map(H_S)
    p0 = OP.reindex(list(zip(v.asset, v.tstart))).to_numpy()
    p1 = CLS.reindex(list(zip(v.asset, v.tstart + hs - 900))).to_numpy()
    sig = VOL.reindex(list(zip(v.asset, v.tstart))).to_numpy() * np.sqrt(hs / 900)
    s = np.where(v.side_up == 1, 1.0, -1.0)
    v['xraw'] = s * np.log(p1 / p0)
    v['z'] = v.xraw / sig
    return v[np.isfinite(v.z) & (v.xraw != 0)]


for k in ['UD_A', 'UD_C']:
    v = running(S[k])
    for lab, sp in [('M1 scheduled end (round 3)', spec(k)), ('M0 actual resolution (round 2)', SP_OLD)]:
        for y in ['ret1d', 'ret7d', 'ret30d', 'l_usd30']:
            r = ols2(v, y, **sp)
            out.append(dict(part='rd regression band', sample=k, timing=lab, spec='regression, same rows', **r))
    log(k, 'rd band done')

out = pd.DataFrame(out)
out.to_csv(f'{R2RES}/R3_ud_extras.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(out[['part', 'sample', 'timing', 'spec', 'y', 'b', 'se', 'p', 'n']].round(4).to_string())
log('done')
