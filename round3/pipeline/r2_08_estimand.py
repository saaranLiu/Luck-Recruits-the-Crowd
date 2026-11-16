"""T7: estimands.  Stratified ATE and ATT versus the OLS (variance-weighted) estimate, and the net activity increment.

Strata: up/down pbin x cell (cell = asset x window length x month as in round 1); sports pbin x month.  Within a
stratum the win is as good as random under the design assumption, so tau_c = mean(Y|win) - mean(Y|lose).
  ATE = sum_c n_c tau_c / sum_c n_c,  ATT = sum_c n1_c tau_c / sum_c n1_c,  VW = sum_c n_c p_c(1-p_c) tau_c / sum (...)
(VW is what OLS with stratum dummies estimates).  Strata without both winners and losers are dropped; the dropped
share of wallets is reported.
Net increment (common 90-day cohort, same window): extra = ATT x n_winners = expected number of additional active
wallets because winners won rather than lost; share = extra / n_active (all newcomers active in the window).  This
is a net counterfactual contribution to the active count, not the number of individuals whose behaviour changed.
Cluster bootstrap (markets for up/down, games for sports), 500 draws.
"""
from r2_common import *
import numpy as np, pandas as pd
from r2_samples import load_r2, samples_r2, spec, ols2

B = 500
rng = np.random.default_rng(SEED + 7)
d = load_r2()
S = samples_r2(d)
KEYS = ['UD_A', 'SP_S1', 'SP_S0']
YS = ['ret7d', 'act22_30', 'act61_90', 'ret30d']
CL = {'UD': 'cid', 'SP': 'event'}


def strata(v, k):
    return (v.pbin.astype(str) + '|' + (v.cell.astype(str) if k[:2] == 'UD' else v.month.astype(str)))


def estimates(c, win, y, wt, C):
    n1 = np.bincount(c, wt * win, C); n0 = np.bincount(c, wt * (1 - win), C)
    s1 = np.bincount(c, wt * win * y, C); s0 = np.bincount(c, wt * (1 - win) * y, C)
    ok = (n1 > 0) & (n0 > 0)
    tau = np.where(ok, s1 / np.where(n1 > 0, n1, 1) - s0 / np.where(n0 > 0, n0, 1), 0)
    n = n1 + n0; p = np.where(n > 0, n1 / np.where(n > 0, n, 1), 0)
    ate = (n * tau)[ok].sum() / n[ok].sum()
    att = (n1 * tau)[ok].sum() / n1[ok].sum()
    vw = (n * p * (1 - p) * tau)[ok].sum() / (n * p * (1 - p))[ok].sum()
    n_act = (wt * y).sum(); n_win = (wt * win).sum()
    extra = att * n_win
    return dict(ate=ate, att=att, vw=vw, extra=extra, share=extra / n_act, n_active=n_act, n_win=n_win,
                dropped=1 - n[ok].sum() / n.sum())


rows = []
for k in KEYS:
    v0 = S[k]
    v0 = v0[v0.ok90].copy()
    v0['stratum'] = strata(v0, k)
    for y in YS:
        v = v0.dropna(subset=[y]).reset_index(drop=True)
        c = pd.factorize(v.stratum)[0]; C = c.max() + 1
        win = v.win.to_numpy(float); Y = v[y].to_numpy(float)
        est = estimates(c, win, Y, np.ones(len(v)), C)
        g = pd.factorize(v[CL[k[:2]]])[0]; G = g.max() + 1
        bs = []
        for _ in range(B):
            wt = np.bincount(rng.integers(0, G, G), minlength=G)[g].astype(float)
            bs.append(estimates(c, win, Y, wt, C))
        bs = pd.DataFrame(bs)
        ols_main = ols2(v, y, **spec(k), cluster=CL[k[:2]])
        ols_strat = ols2(v, y, ctrl=(), cat=('stratum',), cluster=CL[k[:2]])
        r = dict(sample=k, y=y, frame='common 90d', n=len(v), n_strata=C, dropped_share=est['dropped'],
                 n_winners=est['n_win'], n_active=est['n_active'], base_rate_losers=Y[win == 0].mean(),
                 ols_main=ols_main['b'], ols_main_se=ols_main['se'], ols_strata_only=ols_strat['b'])
        for m in ['ate', 'att', 'vw', 'extra', 'share']:
            r[m] = est[m]; r[f'{m}_se'] = bs[m].std(); r[f'{m}_lo'] = bs[m].quantile(.025); r[f'{m}_hi'] = bs[m].quantile(.975)
        rows.append(r)
        log(k, y, {m: round(est[m], 4) for m in ['ate', 'att', 'vw', 'share']}, 'ols', round(ols_main['b'], 4))
out = pd.DataFrame(rows)
out['note'] = 'share = ATT x n_winners / n_active: net counterfactual contribution of winning to the active count'
out.to_csv(f'{R2RES}/T7_net_activity_share.csv', index=False)
pd.set_option('display.width', 250)
print(out[['sample', 'y', 'n', 'dropped_share', 'ols_main', 'vw', 'ate', 'ate_se', 'att', 'att_se', 'extra', 'share', 'share_lo', 'share_hi']].round(4).to_string())
log('done')
