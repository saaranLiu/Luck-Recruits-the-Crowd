"""Diagnostics for the as-good-as-random assumption.

Up/down: win rates by asset, window, month, and side; out-of-sample prediction of the win from information available
before the first order (order variables, prior Binance returns and volatility, outcomes of earlier windows), with
time-ordered folds that keep all wallets of a market in one fold.
Sports: calibration of the price paid; prediction of the win beyond the price; first order placed before the scheduled
start (metadata close time validated as the start by a game-length gap to resolution)."""
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from prep_ex import *

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
d = load_ex()
S = samples_ex(d)
rng = np.random.default_rng(3)


def cmean(s, y, cl):
    """Mean of y with a cluster-robust 95% interval."""
    s = s.dropna(subset=[y])
    r = sm.OLS(s[y].astype(float), np.ones(len(s))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s[cl])[0]})
    return dict(mean=r.params.iloc[0], lo=r.params.iloc[0] - 1.96 * r.bse.iloc[0], hi=r.params.iloc[0] + 1.96 * r.bse.iloc[0],
                n=len(s), n_cl=s[cl].nunique())


def wald(s, y, cols, cl):
    """Cluster-robust joint test that y does not vary across the categories of cols."""
    s = s.dropna(subset=[y])
    X = sm.add_constant(pd.get_dummies(s[cols].astype(str), drop_first=True, dtype=float))
    r = sm.OLS(s[y].astype(float), X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s[cl])[0]})
    R = np.eye(X.shape[1])[1:]
    t = r.wald_test(R, scalar=True)
    return dict(stat=float(t.statistic), df=R.shape[0], p=float(t.pvalue))


# ---- 1. up/down win rates by cell ---------------------------------------------------------------------------------
rows = []
for k in ['UD_A', 'UD_C']:
    v = S[k].assign(side=lambda x: np.where(x.side_up == 1, 'up', 'down'))
    rows.append(dict(sample=k, dim='all', level='all', **cmean(v, 'win', 'cid')))
    for dim in ['asset', 'hz', 'month', 'side']:
        for lev, g in v.groupby(dim):
            if len(g) >= 50: rows.append(dict(sample=k, dim=dim, level=lev, **cmean(g, 'win', 'cid')))
        rows.append(dict(sample=k, dim=dim, level='joint test', **wald(v[v.groupby(dim)[dim].transform('size') >= 50], 'win', [dim], 'cid')))
    # market-level: share of windows that ended up, by asset (the coin itself, independent of who bet)
    m = v.drop_duplicates('cid')
    for lev, g in m.groupby('asset'):
        if len(g) >= 50: rows.append(dict(sample=k, dim='market_up_by_asset', level=lev, **cmean(g, 'up', 'cid')))
win_tab = pd.DataFrame(rows)
win_tab.to_csv(f'{RES}/rand_ud_winrates.csv', index=False)
print(win_tab.round(4).to_string())

# ---- 2. up/down: predict the win from pre-bet information -----------------------------------------------------------
k15 = pd.read_parquet(f'{WORK}/klines/binance_15m.parquet')
k15['asset'] = k15.sym.str.replace('USDT', '').str.lower()
k15 = k15.sort_values(['asset', 't'])
k15['lr'] = k15.groupby('asset').close.transform(lambda s: np.log(s).diff())
k15['vol24'] = k15.groupby('asset').lr.transform(lambda s: s.rolling(96).std())
CL = k15.set_index(['asset', 't']).close
VOL = k15.set_index(['asset', 't']).vol24
ud_series = pd.read_parquet(f'{WORK}/ud_series.parquet')
ud_series['tend'] = ud_series.tstart + ud_series.hz.map(H_S)


def ud_features(v):
    v = v.copy()
    s = np.where(v.side_up == 1, 1.0, -1.0)
    tl = (v.ts0 // 900) * 900 - 900                      # open time of the last 15-min candle closed before the order
    c0 = CL.reindex(list(zip(v.asset, tl))).to_numpy()
    for lab, h in [('r15m', 1), ('r1h', 4), ('r4h', 16), ('r24h', 96)]:
        ch = CL.reindex(list(zip(v.asset, tl - 900 * h))).to_numpy()
        v[lab] = s * np.log(c0 / ch)
    v['vol24'] = VOL.reindex(list(zip(v.asset, tl))).to_numpy()
    # outcomes of the same asset and window length that had ended before the order (signed by the side bet)
    us = ud_series.sort_values('tend')
    prev = []
    for (a, h), g in v.groupby(['asset', 'hz']):
        u = us[(us.asset == a) & (us.hz == h)]
        te, up = u.tend.to_numpy(), u.up.to_numpy(float)
        cs = np.concatenate([[0], np.cumsum(up)])
        i = np.searchsorted(te, g.ts0.to_numpy(), side='right')       # windows with tend <= ts0
        last1 = np.where(i >= 1, up[np.clip(i - 1, 0, None)], np.nan)
        last4 = np.where(i >= 4, (cs[i] - cs[np.clip(i - 4, 0, None)]) / 4, np.nan)
        prev.append(pd.DataFrame({'prev1': last1, 'prev4': last4}, index=g.index))
    prev = pd.concat(prev).reindex(v.index)
    v['prev1'] = s * (2 * prev.prev1 - 1); v['prev4'] = s * (2 * prev.prev4 - 1)
    v['llead'] = np.log1p(v.lead.clip(lower=0))
    return v


NUM_UD = ['p0', 'lstake', 'llead', 'lfills0', 'hour', 'dow', 'side_up', 'r15m', 'r1h', 'r4h', 'r24h', 'vol24', 'prev1', 'prev4']
CAT_UD = ['asset', 'hz']


def oos_predict(v, num, cat, time_col, base_cols=None, n_blocks=6):
    """Forward-chaining out-of-sample predictions.  Markets are ordered by time and cut into n_blocks contiguous blocks;
    for block b >= 2 the models are trained on all earlier blocks.  Returns test rows with predictions."""
    v = v.copy()
    mt = v.groupby('cid')[time_col].min().sort_values()
    blk = pd.Series(np.floor(np.arange(len(mt)) / len(mt) * n_blocks).astype(int), index=mt.index)
    v['blk'] = v.cid.map(blk)
    X = pd.concat([v[num].astype(float), pd.get_dummies(v[cat].astype(str), dtype=float)], axis=1)
    X = X.fillna(X.median())
    out = []
    for b in range(1, n_blocks):
        tr, te = v.blk < b, v.blk == b
        y_tr = v.win[tr].to_numpy()
        res = pd.DataFrame({'cid': v.cid[te], 'day': v.day[te], 'win': v.win[te], 'blk': b}, index=v.index[te])
        res['p_const'] = y_tr.mean()
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y_tr)
        res['p_logit'] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200,
                                            l2_regularization=5.0, random_state=0)
        gb.fit(X[tr], y_tr)
        res['p_gbm'] = gb.predict_proba(X[te])[:, 1]
        if base_cols:
            Xb = v[base_cols].astype(float)
            lb = LogisticRegression(C=1e6, max_iter=2000).fit(Xb[tr], y_tr)
            res['p_base'] = lb.predict_proba(Xb[te])[:, 1]
        out.append(res)
    return pd.concat(out)


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score(res, ref, models, cl='cid', B=500):
    """Out-of-sample log-loss gain over the reference model (in % of the reference log loss), AUC, accuracy;
    95% intervals from a bootstrap that resamples clusters (markets or days)."""
    out = []
    g = pd.factorize(res[cl])[0]; G = g.max() + 1
    y = res.win.to_numpy(float)
    l_ref = ll(y, res[ref].to_numpy())
    for mname in models:
        p = res[mname].to_numpy()
        l_m = ll(y, p)
        gain = l_ref - l_m
        sg, sr = np.bincount(g, gain, G), np.bincount(g, l_ref, G)
        auc = roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5
        bs_gain, bs_auc = [], []
        for _ in range(B):
            wts = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
            bs_gain.append(100 * (wts @ sg) / (wts @ sr))
            ww = wts[g]
            bs_auc.append(roc_auc_score(y, p, sample_weight=ww) if ww.sum() > 0 else np.nan)
        out.append(dict(model=mname, ref=ref, cluster=cl, n=len(res), n_cl=G, ll_gain_pct=100 * gain.sum() / l_ref.sum(),
                        ll_lo=np.percentile(bs_gain, 2.5), ll_hi=np.percentile(bs_gain, 97.5), auc=auc,
                        auc_lo=np.nanpercentile(bs_auc, 2.5), auc_hi=np.nanpercentile(bs_auc, 97.5),
                        acc=((p > 0.5) == y).mean()))
    return out


pred = []
for k in ['UD_A', 'UD_C']:
    v = ud_features(S[k][S[k].asset.isin(['btc', 'eth', 'sol', 'xrp', 'bnb', 'doge'])])
    v.to_parquet(f'{WORK}/{k}_features.parquet')
    res = oos_predict(v, NUM_UD, CAT_UD, 'tstart')
    for cl in ['cid', 'day']:
        for r in score(res, 'p_const', ['p_logit', 'p_gbm'], cl=cl):
            pred.append(dict(sample=k, **r))
    # in-sample association of each pre-bet feature with the win, conditional on the design cells
    for f in ['r15m', 'r1h', 'r4h', 'r24h', 'prev1', 'prev4', 'vol24']:
        vv = v.assign(fz=(v[f] - v[f].mean()) / v[f].std())
        r = ols2(vv, 'win', x='fz', ctrl=('lstake', 'ldelay'), cat=('pbin', 'cell', 'hour'))
        if r: pred.append(dict(sample=k, model=f'insample_{f}', **{kk: r[kk] for kk in ('b', 'se', 'p', 'n')}))
# ---- 3. sports: calibration and prediction beyond the price ---------------------------------------------------------
cal = []
for k in ['SP_A', 'SP_C']:
    v = S[k].copy()
    v['ex'] = v.win - v.p0
    v['yr'] = v.month.str[:4]
    v['lead_bin'] = pd.cut(v.lead / 3600, [6, 12, 24, 72, np.inf], right=False).astype(str)
    v['lg'] = np.where(v.league.isin(v.league.value_counts().index[:8]), v.league, 'other')
    cal.append(dict(sample=k, dim='all', level='all', **cmean(v, 'ex', 'event')))
    for dim in ['pbin', 'lg', 'yr', 'lead_bin', 'mtype']:
        for lev, g in v.groupby(dim):
            if len(g) >= 200: cal.append(dict(sample=k, dim=dim, level=str(lev), p_mean=g.p0.mean(), win_mean=g.win.mean(),
                                              **cmean(g, 'ex', 'event')))
        cal.append(dict(sample=k, dim=dim, level='joint test', **wald(v[v.groupby(dim)[dim].transform('size') >= 200], 'ex', [dim], 'event')))
    # calibration regression: win on price paid
    r = sm.OLS(v.win.astype(float), sm.add_constant(v.p0)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(v.event)[0]})
    cal.append(dict(sample=k, dim='calib_slope', level='p0', mean=r.params['p0'], lo=r.params['p0'] - 1.96 * r.bse['p0'],
                    hi=r.params['p0'] + 1.96 * r.bse['p0'], n=len(v)))
    # prediction beyond the price
    v['side_ref'] = (v.D0 == 1).astype(float)
    v['llead'] = np.log(v.lead)
    v['logit_p'] = np.log(v.p0 / (1 - v.p0))
    res = oos_predict(v, ['logit_p', 'lstake', 'llead', 'lfills0', 'hour', 'dow', 'side_ref'], ['lg', 'mtype'], 'ts0',
                      base_cols=['logit_p'])
    for cl in ['event', 'day']:
        res['event'] = v.loc[res.index, 'event']
        for r in score(res, 'p_base', ['p_logit', 'p_gbm'], cl=cl):
            pred.append(dict(sample=k, **r))
        for r in score(res, 'p_const', ['p_base'], cl=cl):
            pred.append(dict(sample=k, **r))
cal = pd.DataFrame(cal)
cal.to_csv(f'{RES}/rand_sp_calibration.csv', index=False)
print(cal.round(4).to_string())
pred = pd.DataFrame(pred)
pred.to_csv(f'{RES}/rand_prediction.csv', index=False)
print(pred.round(4).to_string())

# ---- 4. sports: was the first order placed before the game started? ------------------------------------------------
st = []
for k in ['SP_A', 'SP_C']:
    v = S[k]
    ok = v[v.start_ok]
    st.append(dict(sample=k, n=len(v), share_start_known=v.start_ok.mean(), share_pre_start=ok.pre_start.mean(),
                   share_pre_start_1h=(ok.lead_start_h >= 1).mean(), lead_start_med_h=ok.lead_start_h.median(),
                   share_moneyline=(v.mtype == 'moneyline').mean()))
    for lab, s in [('start_known_pre', v[v.start_ok & (v.pre_start == 1)]), ('start_known_post', v[v.start_ok & (v.pre_start == 0)]),
                   ('start_unknown', v[~v.start_ok]), ('moneyline', v[v.mtype == 'moneyline'])]:
        for y in ['ret7d', 'ret30d']:
            r = ols2(s, y, **spec(k), cluster='event')
            if r: st.append(dict(sample=k, subset=lab, **r))
    lg = v[~v.start_ok].league.value_counts(normalize=True).head(8)
    st.append(dict(sample=k, subset='unknown_league_mix', y=str(lg.round(3).to_dict())))
st = pd.DataFrame(st)
st.to_csv(f'{RES}/rand_sp_start.csv', index=False)
print(st.round(4).to_string())
log('done')
