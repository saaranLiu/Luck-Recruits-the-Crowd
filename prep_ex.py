"""Derived variables and nested samples for the ex-ante design (work/exante.parquet, built by 12_build_exante.py).

A  ex-ante sample: first transaction is aggressive, in one market, one side; first-order price in [0.3, 0.7]; placed
   before the window opens (up/down) or at least six hours before resolution (sports).  Nothing after the first order
   is used for selection.  Treatment = the leg bought by the first order wins.
B  A and the wallet neither reduces nor hedges the first position nor trades it passively before resolution.
C  B and the wallet trades in no other market before resolution (the restriction of the earlier draft).
"""
import numpy as np, pandas as pd
from common import *
from prep import ols, design

H_S = {'5m': 300, '15m': 900, '1h': 3600, '4h': 14400}


def load_ex():
    con = connect(threads=32, mem='200GB')
    d = con.sql(f"SELECT * FROM '{WORK}/exante.parquet'").df()
    d['lstake'] = np.log(d.stake0)
    d['pi'] = d.p0
    d['pbin'] = np.floor(d.p0 * 20).astype(int)
    t = pd.to_datetime(d.ts0, unit='s')
    d['month'] = t.dt.to_period('M').astype(str); d['hour'] = t.dt.hour; d['dow'] = t.dt.dayofweek
    d['day'] = (d.ts0 // DAY).astype(int); d['week'] = (d.ts0 // (7 * DAY)).astype(int)
    d['ldelay'] = np.log1p(d.res_ts - d.ts0)
    d['lfills0'] = np.log(d.nf0)
    d['tend'] = d.tstart + d.hz.map(H_S)
    d['lead'] = np.where(d.grp == 'UpDown', d.tstart - d.ts0, d.res_ts - d.ts0)
    d['cell'] = np.where(d.grp == 'UpDown', d.month + '_' + d.hz.fillna('na') + '_' + d.side_up.fillna(-1).astype(int).astype(str), d.month)
    # cumulative return by horizon and activity in disjoint periods after resolution (missing if censored)
    for h, H in [('1d', 1), ('7d', 7), ('14d', 14), ('30d', 30), ('60d', 60), ('90d', 90)]:
        ok = d.res_ts + H * DAY <= T_END
        d[f'ret{h}'] = np.where(ok, (d[f'n_{h}'].fillna(0) > 0).astype(float), np.nan)
    for a, b in [(8, 14), (15, 21), (22, 30), (31, 60), (61, 90)]:
        ok = d.res_ts + b * DAY <= T_END
        d[f'act{a}_{b}'] = np.where(ok, (d[f'n_w{a}_{b}'].fillna(0) > 0).astype(float), np.nan)
    d['act0_7'] = d.ret7d
    ok30, ok90 = d.res_ts + 30 * DAY <= T_END, d.res_ts + 90 * DAY <= T_END
    d['ok90'] = ok90
    d['days30'] = np.where(ok30, d.days_30.fillna(0), np.nan)
    d['weeks90'] = np.where(ok90, d.weeks_90.fillna(0), np.nan)
    d['l_usd30'] = np.where(ok30, np.log1p(d.usd_30d.fillna(0)), np.nan)
    d['l_n30'] = np.where(ok30, np.log1p(d.n_30d.fillna(0)), np.nan)
    # intermediate behaviour between the first order and resolution
    d['b_reduce'] = (d.n_opp > 0).astype(float)
    d['b_exit'] = (d.net_sh / d.gross_sh < 0.05).astype(float)
    d['b_maker'] = (d.n_maker1 > 0).astype(float)
    d['b_add'] = (d.n_add > 0).astype(float)
    d['b_gap'] = (d.n_gap > 0).astype(float)
    d['held'] = ((d.n_opp == 0) & (d.n_maker1 == 0)).astype(float)
    # quality of later aggressive trades within 90 days (resolved markets)
    d['has_q'] = np.where(ok90, (d.q_n.fillna(0) > 0).astype(float), np.nan)
    d['q_ret'] = d.q_pnl / d.q_usd
    for c in ['q_edge_usd', 'q_edge', 'q_ret', 'q_hit']:
        d[c] = np.where(ok90 & (d.q_n.fillna(0) > 0), d[c], np.nan)
    # sports: league, game (event), market type, scheduled start from metadata where it is validated
    sp = d.grp == 'SportsML'
    d['league'] = np.where(sp, d.slug.str.split('-').str[0], d.asset)
    ev = d.slug.str.extract(r'^(.*?-\d{4}-\d{2}-\d{2})')[0]
    d['event'] = np.where(sp, ev.fillna(d.slug), d.cid)
    d['mtype'] = np.where(d.slug.str.match(r'^.*-\d{4}-\d{2}-\d{2}$'), 'moneyline',
                          np.where(d.slug.str.contains(r'-\d{4}-\d{2}-\d{2}-'), 'other', 'nodate'))
    gap = (d.res_ts - d.close_ts) / 3600
    d['start_ok'] = sp & gap.between(1.5, 12)        # close time = scheduled start (game-length gap to resolution)
    d['pre_start'] = np.where(d.start_ok, (d.ts0 < d.close_ts).astype(float), np.nan)
    d['lead_start_h'] = np.where(d.start_ok, (d.close_ts - d.ts0) / 3600, np.nan)
    return d


def samples_ex(d):
    mid = d.p0.between(0.3, 0.7)
    ud = mid & (d.grp == 'UpDown') & (d.ts0 < d.tstart)
    sp = mid & (d.grp == 'SportsML') & (d.res_ts - d.ts0 >= 6 * 3600)
    held = d.held == 1
    C = held & (d.n_gap == 0)
    return {'UD_A': d[ud], 'UD_B': d[ud & held], 'UD_C': d[ud & C],
            'SP_A': d[sp], 'SP_B': d[sp & held], 'SP_C': d[sp & C]}


SPEC_EX = {
    'UD': dict(ctrl=('lstake', 'ldelay'), cat=('pbin', 'cell', 'hour')),
    'SP': dict(ctrl=('lstake',), cat=('pbin', 'month', 'hour')),
}


def spec(k):
    return SPEC_EX[k[:2]]


def ols2(df, y, x='win', ctrl=('lstake',), cat=('pbin', 'month', 'hour'), cluster='cid', w=None, extra=(), fe=None):
    """OLS (or WLS with weights column w) of y on x and controls; cluster may be one column or a pair (two-way)."""
    import statsmodels.api as sm
    cl = [cluster] if isinstance(cluster, str) else list(cluster)
    cols = [y, x, 'cid'] + list(ctrl) + list(cat) + list(extra) + cl + ([w] if w else []) + ([fe] if fe else [])
    s = df[list(dict.fromkeys(cols))].dropna()
    if fe:
        s = s[s.groupby(fe)[x].transform('nunique') > 1]
    if len(s) < 100:
        return None
    X = design(s, x, ctrl, cat, extra)
    Y = s[y].astype(float)
    if fe:
        X = X - X.groupby(s[fe]).transform('mean'); Y = Y - Y.groupby(s[fe]).transform('mean')
        X = X.loc[:, X.std() > 1e-12]
    else:
        X = sm.add_constant(X)
    g = pd.factorize(s[cl[0]])[0] if len(cl) == 1 else np.column_stack([pd.factorize(s[c])[0] for c in cl])
    mod = sm.WLS(Y, X, weights=s[w].astype(float)) if w else sm.OLS(Y, X)
    r = mod.fit(cov_type='cluster', cov_kwds={'groups': g})
    b, se = r.params[x], r.bse[x]
    return dict(y=y, b=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se, p=r.pvalues[x], n=len(s), n_mkt=s.cid.nunique(),
                n_cl=len(np.unique(g if g.ndim == 1 else g[:, 0])), base=s.loc[s[x] == 0, y].mean(), win_rate=s[x].mean())
