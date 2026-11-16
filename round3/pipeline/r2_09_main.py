"""Main-text results regenerated on the round-2 samples (up/down A/B/C; sports S1 with B/C and S0 as comparison).
Round 3: up/down timing control = log time to the scheduled window end; sports S1 lead time = time to the scheduled
start (balance, prediction features, and the rich-control lead bins); S0 keeps the time to resolution.

1. Sample table: size, balance, returning (own window), staying (common 90-day frame).  Up/down SEs clustered by
   market, sports by game.
2. Behaviour between the first order and the resolution (sample A / S1 / S0).
3. Dynamic effects on the common 90-day frame (Figure 2).
4. Sports S1 diagnostics: calibration and prediction beyond the price (forward-chaining folds, game bootstrap).
5. Inference rows not in T5: SE clustered by market for sports, rich pre-bet controls, game fixed effects.
6. Heterogeneity by first stake; sports timing split into wait before the scheduled start and start-to-resolution.
"""
from r2_common import *
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from r2_samples import load_r2, samples_r2, spec, ols2

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
rng = np.random.default_rng(SEED + 9)
d = load_r2()
S = samples_r2(d)
CL = lambda k: 'cid' if k[:2] == 'UD' else 'event'
TAB = ['UD_A', 'UD_B', 'UD_C', 'SP_S1', 'SP_S1B', 'SP_S1C', 'SP_S0']
BASE = {'UD_A': 'UD_A', 'UD_B': 'UD_A', 'UD_C': 'UD_A', 'SP_S1': 'SP_S1', 'SP_S1B': 'SP_S1', 'SP_S1C': 'SP_S1', 'SP_S0': 'SP_S0'}

# ---- 1. sample table ----------------------------------------------------------------------------------------------
rows = []
for k in TAB:
    v = S[k].assign(llead=lambda x: np.log1p((x.lead_s if k.startswith('SP_S1') else x.lead).clip(lower=0)))
    sp = spec(k); cl = CL(k)
    rows += [dict(sample=k, stat='n', b=len(v)), dict(sample=k, stat='markets', b=v.cid.nunique()),
             dict(sample=k, stat='games', b=v.event.nunique() if k[:2] == 'SP' else np.nan),
             dict(sample=k, stat='share_of_A', b=len(v) / len(S[BASE[k]])), dict(sample=k, stat='win_rate', b=v.win.mean()),
             dict(sample=k, stat='n_common90', b=int(v.ok90.sum())), dict(sample=k, stat='stake_median', b=v.stake0.median()),
             dict(sample=k, stat='first', b=pd.to_datetime(v.ts0.min(), unit='s').strftime('%Y-%m')),
             dict(sample=k, stat='last', b=pd.to_datetime(v.ts0.max(), unit='s').strftime('%Y-%m'))]
    if k[:2] == 'SP':
        lead_start = (v.ind_start - v.ts0) / 3600
        rows.append(dict(sample=k, stat='lead_start_med_h', b=lead_start.median()))
        rows.append(dict(sample=k, stat='delay_med_h', b=((v.res_ts - v.ts0) / 3600).median()))
        rows += [dict(sample=k, stat=f'league_{lg}', b=s) for lg, s in v.league.value_counts(normalize=True).head(5).items()]
    for y in ['pi', 'lstake', 'llead']:
        cat = ('pbin', 'cell') if k[:2] == 'UD' else ('pbin', 'month')
        rows.append(dict(sample=k, stat=f'bal_{y}', **ols2(v, y, ctrl=(), cat=cat, cluster=cl)))
    for y in ['ret1d', 'ret7d', 'ret30d', 'ret90d']:
        rows.append(dict(sample=k, stat=f'eff_{y}', frame='own window', **ols2(v, y, **sp, cluster=cl)))
    r = ols2(v, 'ret7d', ctrl=sp['ctrl'], cat=('pbin',), fe='cid', cluster=cl)
    if r: rows.append(dict(sample=k, stat='mfe_ret7d', frame='own window', **r))
    c90 = v[v.ok90]
    for y in ['ret7d', 'ret30d', 'act22_30', 'act61_90', 'days30', 'weeks90', 'l_usd30']:
        rows.append(dict(sample=k, stat=f'c90_{y}', frame='common 90d', **ols2(c90, y, **sp, cluster=cl)))
    log(k, 'sample table')
tab = pd.DataFrame(rows)
tab.to_csv(f'{R2RES}/R_samples.csv', index=False)

# ---- 2. intermediate behaviour -------------------------------------------------------------------------------------
beh = []
for k in ['UD_A', 'SP_S1', 'SP_S0']:
    v = S[k].copy()
    v['inB'] = v.held.astype(float)
    v['inC'] = ((v.held == 1) & (v.n_gap == 0)).astype(float)
    for y in ['b_reduce', 'b_exit', 'b_add', 'b_maker', 'b_gap', 'inB', 'inC']:
        beh.append(dict(sample=k, **ols2(v, y, **spec(k), cluster=CL(k)), mean=v[y].mean()))
    red = v[v.n_opp > 0]
    if k == 'UD_A':
        beh.append(dict(sample=k, y='reduce_after_window_open', b=(red.t_first_opp >= red.tstart).mean()))
    else:
        beh.append(dict(sample=k, y='reduce_after_sched_start', b=(red.t_first_opp >= red.ind_start).mean()))
beh = pd.DataFrame(beh)
beh.to_csv(f'{R2RES}/R_intermediate.csv', index=False)
log('intermediate')

# ---- 3. dynamics on the common 90-day frame ------------------------------------------------------------------------
CUM = ['ret1d', 'ret7d', 'ret14d', 'ret30d', 'ret60d', 'ret90d']
PER = ['act0_7', 'act8_14', 'act15_21', 'act22_30', 'act31_60', 'act61_90']
dyn = []
for k in ['UD_A', 'UD_C', 'SP_S1', 'SP_S1C', 'SP_S0']:
    v = S[k][S[k].ok90]
    for y in CUM + PER:
        dyn.append(dict(sample=k, kind='cum' if y in CUM else 'per', **ols2(v, y, **spec(k), cluster=CL(k))))
dyn = pd.DataFrame(dyn)
dyn.to_csv(f'{R2RES}/R_dynamic.csv', index=False)
log('dynamics')


# ---- 4. sports S1 diagnostics --------------------------------------------------------------------------------------
def cmean(s, y, cl):
    s = s.dropna(subset=[y])
    r = sm.OLS(s[y].astype(float), np.ones(len(s))).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s[cl])[0]})
    return dict(mean=r.params.iloc[0], lo=r.params.iloc[0] - 1.96 * r.bse.iloc[0], hi=r.params.iloc[0] + 1.96 * r.bse.iloc[0], n=len(s))


def wald(s, y, cols, cl):
    s = s.dropna(subset=[y])
    X = sm.add_constant(pd.get_dummies(s[cols].astype(str), drop_first=True, dtype=float))
    r = sm.OLS(s[y].astype(float), X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(s[cl])[0]})
    t = r.wald_test(np.eye(X.shape[1])[1:], scalar=True)
    return dict(stat=float(t.statistic), p=float(t.pvalue))


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


cal, pred = [], []
for k in ['SP_S1', 'SP_S0']:
    v = S[k].copy()
    v['ex'] = v.win - v.p0
    v['yr'] = v.month.str[:4]
    v['lg'] = np.where(v.league.isin(v.league.value_counts().index[:8]), v.league, 'other')
    v['lead_bin'] = pd.cut(((v.ind_start - v.ts0) if k == 'SP_S1' else (v.res_ts - v.ts0)) / 3600, [0, 6, 12, 24, 72, np.inf], right=False).astype(str)
    cal.append(dict(sample=k, dim='all', level='all', **cmean(v, 'ex', 'event')))
    for dim in ['pbin', 'lg', 'yr', 'lead_bin', 'mtype']:
        vv = v[v.groupby(dim)[dim].transform('size') >= 200]
        if vv[dim].nunique() > 1:
            cal.append(dict(sample=k, dim=dim, level='joint test', **wald(vv, 'ex', [dim], 'event')))
    r = sm.OLS(v.win.astype(float), sm.add_constant(v.p0)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(v.event)[0]})
    cal.append(dict(sample=k, dim='calib_slope', level='p0', mean=r.params['p0'], lo=r.params['p0'] - 1.96 * r.bse['p0'],
                    hi=r.params['p0'] + 1.96 * r.bse['p0'], n=len(v)))
    # forward-chaining prediction beyond the price; games kept within one block
    v['side_ref'] = (v.D0 == 1).astype(float)
    v['llead'] = np.log1p((v.lead_s if k == 'SP_S1' else v.lead).clip(lower=0))
    v['logit_p'] = np.log(v.p0 / (1 - v.p0))
    num = ['logit_p', 'lstake', 'llead', 'lfills0', 'hour', 'dow', 'side_ref']
    gt = v.groupby('event').ts0.min().sort_values()
    blk = pd.Series(np.floor(np.arange(len(gt)) / len(gt) * 6).astype(int), index=gt.index)
    v['blk'] = v.event.map(blk)
    X = pd.concat([v[num].astype(float), pd.get_dummies(v[['lg', 'mtype']].astype(str), dtype=float)], axis=1)
    X = X.fillna(X.median())
    out = []
    for b in range(1, 6):
        tr, te = v.blk < b, v.blk == b
        y_tr = v.win[tr].to_numpy()
        res = pd.DataFrame({'event': v.event[te], 'win': v.win[te]}, index=v.index[te])
        res['p_const'] = y_tr.mean()
        res['p_base'] = LogisticRegression(C=1e6, max_iter=2000).fit(v[['logit_p']][tr], y_tr).predict_proba(v[['logit_p']][te])[:, 1]
        sc = StandardScaler().fit(X[tr])
        res['p_logit'] = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y_tr).predict_proba(sc.transform(X[te]))[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200,
                                            l2_regularization=5.0, random_state=0).fit(X[tr], y_tr)
        res['p_gbm'] = gb.predict_proba(X[te])[:, 1]
        out.append(res)
    res = pd.concat(out)
    g = pd.factorize(res.event)[0]; G = g.max() + 1
    y = res.win.to_numpy(float)
    for ref, models in [('p_const', ['p_base']), ('p_base', ['p_logit', 'p_gbm'])]:
        l_ref = ll(y, res[ref].to_numpy())
        for m in models:
            gain = l_ref - ll(y, res[m].to_numpy())
            sg, sr = np.bincount(g, gain, G), np.bincount(g, l_ref, G)
            bs = [100 * (w @ sg) / (w @ sr) for w in (np.bincount(rng.integers(0, G, G), minlength=G).astype(float) for _ in range(500))]
            pred.append(dict(sample=k, model=m, ref=ref, n=len(res), n_games=G, ll_gain_pct=100 * gain.sum() / l_ref.sum(),
                             ll_lo=np.percentile(bs, 2.5), ll_hi=np.percentile(bs, 97.5), auc=roc_auc_score(y, res[m])))
    log(k, 'diagnostics')
pd.DataFrame(cal).to_csv(f'{R2RES}/R_sp_calibration.csv', index=False)
pd.DataFrame(pred).to_csv(f'{R2RES}/R_sp_prediction.csv', index=False)

# ---- 5. inference rows not covered by T5 ---------------------------------------------------------------------------
rob = []
for k in ['SP_S1', 'SP_S0']:
    v = S[k].copy()
    v['lead_bin'] = pd.cut((v.lead_s if k == 'SP_S1' else v.lead) / 3600, [0, 6, 12, 24, 72, np.inf], right=False).astype(str)
    v['lg'] = np.where(v.league.isin(v.league.value_counts().index[:20]), v.league, 'other')
    v['side_ref'] = (v.D0 == 1).astype(float)
    v['pbin1'] = np.floor(v.p0 * 100).astype(int)
    for y in ['ret7d', 'ret30d']:
        rob.append(dict(sample=k, spec='main, clustered by market', **ols2(v, y, **spec(k), cluster='cid')))
        rob.append(dict(sample=k, spec='rich pre-bet controls', **ols2(v, y, ctrl=('lstake', 'lfills0', 'side_ref'),
                        cat=('pbin1', 'month', 'hour', 'lg', 'lead_bin', 'mtype', 'dow'), cluster='event')))
        r = ols2(v, y, ctrl=('lstake',), cat=('pbin', 'hour'), cluster='event', fe='event')
        if r: rob.append(dict(sample=k, spec='game fixed effects', **r))
pd.DataFrame(rob).to_csv(f'{R2RES}/R_inference_extra.csv', index=False)
log('inference extra')

# ---- 6. heterogeneity ----------------------------------------------------------------------------------------------
BINS = [0, 2, 5, 20, 100, np.inf]; LAB = ['<$2', '$2-5', '$5-20', '$20-100', '>$100']
het = []
for k in ['UD_A', 'SP_S1', 'SP_S0']:
    v = S[k].copy()
    v['sbin'] = pd.cut(v.stake0, BINS, labels=LAB, right=False)
    for q, g in v.groupby('sbin', observed=True):
        r = ols2(g, 'ret7d', **spec(k), cluster=CL(k))
        if r: het.append(dict(sample=k, dim='stake', level=str(q), share=len(g) / len(v), **r))
v = S['SP_S1'].copy()
v['l_pre'] = np.log((v.ind_start - v.ts0) / 3600 + 0.1)
v['l_post'] = np.log((v.res_ts - v.ind_start).clip(lower=600) / 3600)
for c in ['l_pre', 'l_post']:
    v[c + '_c'] = v[c] - v[c].mean(); v['win_' + c] = v.win * v[c + '_c']
for y in ['ret1d', 'ret7d', 'ret30d']:
    for x, other in [('win_l_pre', 'win_l_post'), ('win_l_post', 'win_l_pre')]:
        r = ols2(v, y, x=x, ctrl=('lstake', 'win', 'l_pre_c', 'l_post_c', other), cat=('pbin', 'month', 'hour', 'league'), cluster='event')
        het.append(dict(sample='SP_S1', dim='timing', level=x, pre_med_h=((v.ind_start - v.ts0) / 3600).median(),
                        post_med_h=((v.res_ts - v.ind_start) / 3600).median(), **r))
het = pd.DataFrame(het)
het.to_csv(f'{R2RES}/R_heterogeneity.csv', index=False)

# ---- 7. net shares on the same cohort for every sample in the table (from T7 logic, point estimates) ---------------
print(tab[['sample', 'stat', 'b', 'se', 'n', 'base']].round(4).to_string())
print(beh[['sample', 'y', 'b', 'se', 'mean']].round(4).to_string())
print(dyn[['sample', 'kind', 'y', 'b', 'se', 'n']].round(4).to_string())
print(pd.DataFrame(cal).round(4).to_string()); print(pd.DataFrame(pred).round(4).to_string())
print(pd.DataFrame(rob)[['sample', 'spec', 'y', 'b', 'se', 'n']].round(4).to_string())
print(het[['sample', 'dim', 'level', 'y', 'b', 'se', 'n']].round(4).to_string())
log('done')
