"""Derived variables, analysis samples, and the OLS estimator shared by the estimation scripts."""
import numpy as np, pandas as pd, statsmodels.api as sm
from common import *


def load():
    con = connect()
    d = con.sql(f"SELECT * FROM '{WORK}/firstbet.parquet'").df()
    d['lstake'] = np.log(d.stake)
    d['pbin'] = np.floor(d.pi * 20).astype(int)
    dt = pd.to_datetime(d.ts0, unit='s')
    d['month'] = dt.dt.to_period('M').astype(str); d['hour'] = dt.dt.hour; d['dow'] = dt.dt.dayofweek
    d['ldelay'] = np.log1p(d.res_ts - d.ts1)
    d['lfills'] = np.log(d.n_fills)
    d['lead'] = np.where(d.grp == 'UpDown', d.tstart - d.ts1, d.res_ts - d.ts1)
    d['cell'] = np.where(d.grp == 'UpDown', d.month + '_' + d.hz.fillna('na') + '_' + d.side_up.fillna(-1).astype(int).astype(str), d.month)
    for h, H in [('1d', 1), ('7d', 7), ('30d', 30), ('90d', 90)]:
        ok = d.res_ts + H * DAY <= T_END
        d[f'ret{h}'] = np.where(ok, (d[f'n_{h}'] > 0).astype(float), np.nan)
    ok30 = d.res_ts + 30 * DAY <= T_END
    d['l_n30'] = np.where(ok30, np.log1p(d.n_30d), np.nan)
    d['l_usd30'] = np.where(ok30, np.log1p(d.usd_30d), np.nan)
    d['mk30'] = np.where(ok30, d.mk_30d, np.nan)
    d['ret1h'] = ((d.t_next.notna()) & (d.t_next <= 3600)).astype(float)
    has2 = d.stake2.notna()
    d['lratio2'] = np.where(has2, np.log(d.stake2) - d.lstake, np.nan)
    d['longshot2'] = np.where(has2, (d.pi2 < 0.3).astype(float), np.nan)
    d['pi2c'] = np.where(has2, d.pi2, np.nan)
    d['same_grp2'] = np.where(has2, (d.grp2 == d.grp).astype(float), np.nan)
    d['lgap2'] = np.where(has2, np.log1p((d.ts_next_bet - d.res_ts) / 3600), np.nan)
    ok90 = d.res_ts + 90 * DAY <= T_END
    d['has_q'] = np.where(ok90, (d.q_n.fillna(0) > 0).astype(float), np.nan)
    d['redeemed'] = d.ts_red.notna().astype(float)
    return d


def samples(d):
    base = (d.n_gap == 0)
    mid = d.pi.between(0.3, 0.7)
    return {
        'UD_pre': d[base & mid & (d.grp == 'UpDown') & d.pre_window.fillna(False)],
        'UD_all': d[base & mid & (d.grp == 'UpDown')],
        'SP': d[base & mid & (d.grp == 'SportsML')],
        'SP_pre': d[base & mid & (d.grp == 'SportsML') & (d.res_ts - d.ts1 >= 6 * 3600)],
        'UD_pre46': d[base & d.pi.between(0.4, 0.6) & (d.grp == 'UpDown') & d.pre_window.fillna(False)],
        'YN': d[base & mid & (d.grp == 'YesNo')],
    }


SPEC = {  # design per sample: controls, categorical FE, absorbed FE
    'UD_pre': dict(ctrl=('lstake', 'ldelay'), cat=('pbin', 'cell', 'hour'), fe=None),
    'UD_all': dict(ctrl=('lstake', 'ldelay'), cat=('pbin', 'cell', 'hour'), fe=None),
    'SP': dict(ctrl=('lstake',), cat=('pbin', 'month', 'hour'), fe=None),
    'SP_pre': dict(ctrl=('lstake',), cat=('pbin', 'month', 'hour'), fe=None),
    'UD_pre46': dict(ctrl=('lstake', 'ldelay'), cat=('pbin', 'cell', 'hour'), fe=None),
    'YN': dict(ctrl=('lstake', 'ldelay'), cat=('pbin', 'month', 'hour'), fe=None),
}


def design(s, x, ctrl, cat, extra=()):
    return pd.concat([s[[x] + list(ctrl) + list(extra)].astype(float)] +
                     [pd.get_dummies(s[c].astype(str), prefix=c, drop_first=True, dtype=float) for c in cat], axis=1)


def ols(df, y, x='win', ctrl=('lstake',), cat=('pbin', 'month', 'hour'), fe=None, extra=()):
    """OLS of y on x + controls; optional one-way fixed effect absorbed by within-demeaning; SE clustered by market."""
    cols = [y, x, 'cid'] + list(ctrl) + list(cat) + list(extra) + ([fe] if fe else [])
    s = df[list(dict.fromkeys(cols))].dropna()
    if fe:
        s = s[s.groupby(fe)[x].transform('nunique') > 1]
    if len(s) < 200:
        return None
    X = design(s, x, ctrl, cat, extra)
    Y = s[y].astype(float)
    if fe:
        X = X - X.groupby(s[fe]).transform('mean'); Y = Y - Y.groupby(s[fe]).transform('mean')
        X = X.loc[:, X.std() > 1e-12]
    else:
        X = sm.add_constant(X)
    r = sm.OLS(Y, X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s['cid'])[0]})
    return dict(y=y, b=r.params[x], se=r.bse[x], p=r.pvalues[x], n=len(s), n_mkt=s.cid.nunique(),
                base=s.loc[s[x] == 0, y].mean(), win_rate=s[x].mean())
