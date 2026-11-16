"""T4: later bets.  Q1 conditional comparison with full TOST output; Q2 stratified Lee bounds with documented strata and
cluster bootstrap by market (up/down) or game (sports); Q3 complier mean with its denominator reported.

Samples: up/down A; sports S1 (ex-ante, main) and S0 (round-1 A).  Newcomers with a complete 90-day window.
Metrics (r2_03_quality_build.py): edge_share (main; gross P&L per contract traded, cents), edge_cost (round-1 cost-
weighted edge), ret_turn (net P&L per dollar traded, %), win_leg (share of fills on the eventually winning leg, pp).
Equivalence margins, kept from round 1: 1 cent per contract (one price tick), 2% per dollar traded (one tick at a price
of 0.5), 2 pp winning-leg share.  Sensitivity: 1 pp winning-leg share (one tick of edge at fixed prices).
Significance level 0.05 throughout; TOST p = max of the two one-sided p values.
"""
from r2_common import *
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import brentq
from multiprocessing import Pool
from r2_samples import load_r2, samples_r2, ols2, spec

DELTA = {'edge_share': 0.01, 'edge_cost': 0.01, 'ret_turn': 0.02, 'win_leg': 0.02}
Y_Q = list(DELTA)
CL = {'UD': 'cid', 'SP': 'event'}
B = 500

d = load_r2()
q = pd.read_parquet(f'{R2WORK}/quality_r2.parquet')
d = d.merge(q.rename(columns={'r_win_leg': 'win_leg'}), on='w', how='left')
d['has_q2'] = np.where(d.ok90, (d.r_n.fillna(0) > 0).astype(float), np.nan)
for c in Y_Q:
    d[c] = np.where(d.ok90 & (d.r_n.fillna(0) > 0), d[c], np.nan)
S = samples_r2(d)
KEYS = ['UD_A', 'SP_S1', 'SP_S0']

# ---- coverage ----------------------------------------------------------------------------------------------------------
cov = []
for k in KEYS:
    v = S[k][S[k].ok90]
    g = v[v.has_q2 == 1]
    cov.append(dict(sample=k, n=len(v), share_trade_win=v[v.win == 1].has_q2.mean(), share_trade_lose=v[v.win == 0].has_q2.mean(),
                    unresolved_share=g.r_c_unres.fillna(0).sum() / (g.r_c.sum() + g.r_c_unres.fillna(0).sum()),
                    passive_share=g.r_c_passive.fillna(0).sum() / (g.r_c.sum() + g.r_c_passive.fillna(0).sum()),
                    fee_share=g.r_fee.sum() / g.r_c.sum(),
                    loser_edge_share=g[g.win == 0].edge_share.mean(), loser_edge_cost=g[g.win == 0].edge_cost.mean(),
                    loser_ret=g[g.win == 0].ret_turn.mean(), loser_win_leg=g[g.win == 0].win_leg.mean(),
                    first=pd.to_datetime(v.ts0.min(), unit='s').date(), last=pd.to_datetime(v.ts0.max(), unit='s').date()))
cov = pd.DataFrame(cov)
cov.to_csv(f'{R2RES}/T4_coverage.csv', index=False)
print(cov.round(4).T.to_string())

# ---- Q1 ------------------------------------------------------------------------------------------------------------------
q1 = []
for k in KEYS:
    v = S[k][S[k].ok90 & (S[k].has_q2 == 1)].copy()
    v['wq'] = v.r_c
    for y in Y_Q:
        for wlab, w in [('wallet-equal (main)', None), ('volume-weighted (turnover)', 'wq')]:
            r = ols2(v, y, **spec(k), w=w, cluster=CL[k[:2]])
            for dl in sorted({DELTA[y], 0.01} if y == 'win_leg' else {DELTA[y]}):
                p_lo = 1 - stats.norm.cdf((r['b'] + dl) / r['se'])        # H0: b <= -dl
                p_hi = stats.norm.cdf((r['b'] - dl) / r['se'])            # H0: b >= dl
                pt = max(p_lo, p_hi)
                q1.append(dict(sample=k, metric=y, weight=wlab, margin=dl, margin_role='prespecified' if dl == DELTA[y] else 'sensitivity',
                               b=r['b'], se=r['se'], ci95_lo=r['lo'], ci95_hi=r['hi'], ci90_lo=r['b'] - 1.645 * r['se'],
                               ci90_hi=r['b'] + 1.645 * r['se'], p_diff=r['p'], p_lower=p_lo, p_upper=p_hi, p_tost=pt,
                               decision=('equivalent at 5%' if pt < 0.05 else 'equivalence not established') +
                                        ('; difference significant' if r['p'] < 0.05 else '; difference not significant'),
                               n=r['n'], n_clusters=r['n_cl'], cluster=CL[k[:2]], base=r['base']))
q1 = pd.DataFrame(q1)
q1.to_csv(f'{R2RES}/T4_quality_and_equivalence.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(q1[['sample', 'metric', 'weight', 'margin', 'b', 'se', 'ci90_lo', 'ci90_hi', 'p_tost', 'decision', 'n']].round(4).to_string())


# ---- Q2: Lee bounds ------------------------------------------------------------------------------------------------------
def lee(v, y, strata, min_n=20):
    """Stratified Lee bounds on E[y(1) - y(0)] for always-traders.  Within each stratum the arm with the larger share of
    later traders is trimmed (monotonicity within the stratum); stratum weight = its estimated number of always-traders.
    Strata with fewer than min_n newcomers in either arm are dropped (share reported)."""
    lo_num = hi_num = den = 0.0
    n_drop = n_win_more = n_used = 0
    for _, g in v.groupby(strata):
        w1, w0 = g[g.win == 1], g[g.win == 0]
        if len(w1) < min_n or len(w0) < min_n:
            n_drop += len(g); continue
        p1, p0 = w1.has_q2.mean(), w0.has_q2.mean()
        a1, a0 = w1.loc[w1.has_q2 == 1, y].dropna().to_numpy(), w0.loc[w0.has_q2 == 1, y].dropna().to_numpy()
        if len(a1) < 5 or len(a0) < 5:
            n_drop += len(g); continue
        if p1 >= p0:
            t = (p1 - p0) / p1; a = np.sort(a1); m = int(round(t * len(a))); n_win_more += 1
            lo, hi = a[:len(a) - m].mean() - a0.mean(), a[m:].mean() - a0.mean()
        else:
            t = (p0 - p1) / p0; a = np.sort(a0); m = int(round(t * len(a)))
            lo, hi = a1.mean() - a[m:].mean(), a1.mean() - a[:len(a) - m].mean()
        wgt = len(g) * min(p1, p0)
        lo_num += wgt * lo; hi_num += wgt * hi; den += wgt; n_used += 1
    return lo_num / den, hi_num / den, n_drop / len(v), n_win_more / max(n_used, 1), n_used


def boot_idx(v, cl, B, seed):
    rng = np.random.default_rng(seed)
    m = pd.factorize(v[cl])[0]; M = m.max() + 1
    order = np.argsort(m, kind='stable'); bounds = np.searchsorted(m[order], np.arange(M + 1))
    for _ in range(B):
        draw = rng.integers(0, M, M)
        yield np.concatenate([order[bounds[i]:bounds[i + 1]] for i in draw])


STRATA = {'UD': {'side x window (round 1)': ['side_up', 'hz'], 'side x window x month (finer)': ['side_up', 'hz', 'month']},
          'SP': {'price bin (round 1)': ['pbin'], 'price bin x month (finer)': ['pbin', 'month']}}


def job(args):
    k, y, slab = args
    v = S[k][S[k].ok90].copy()
    st = STRATA[k[:2]][slab]
    lo, hi, drop, wm, nu = lee(v, y, st)
    bs = np.array([lee(v.iloc[ix], y, st)[:2] for ix in boot_idx(v, CL[k[:2]], B, SEED + hash((k, y, slab)) % 1000)])
    s_lo, s_hi = bs[:, 0].std(), bs[:, 1].std()
    r_ = max(hi - lo, 0) / max(s_lo, s_hi)
    c = brentq(lambda c: stats.norm.cdf(c + r_) - stats.norm.cdf(-c) - 0.95, 1.0, 2.5)
    return dict(sample=k, metric=y, strata=slab, lee_lo=lo, lee_hi=hi, im95_lo=lo - c * s_lo, im95_hi=hi + c * s_hi,
                se_lo=s_lo, se_hi=s_hi, share_dropped_sparse=drop, share_strata_winners_trade_more=wm, n_strata=nu,
                p_trade_win=v[v.win == 1].has_q2.mean(), p_trade_lose=v[v.win == 0].has_q2.mean(), n=len(v),
                bootstrap=f'{B} draws of {CL[k[:2]]}')


jobs = [(k, y, s) for k in KEYS for y in Y_Q for s in STRATA[k[:2]]]
with Pool(min(len(jobs), 32)) as pool:
    q2 = pd.DataFrame(pool.map(job, jobs))
q2.to_csv(f'{R2RES}/T4_lee.csv', index=False)
print(q2.round(4).to_string())


# ---- Q3: compliers -------------------------------------------------------------------------------------------------------
def complier(v, y):
    num = den = 0.0
    for _, g in v.groupby('pbin'):
        w1, w0 = g[g.win == 1], g[g.win == 0]
        if len(w1) < 20 or len(w0) < 20:
            continue
        p1, p0 = w1.has_q2.mean(), w0.has_q2.mean()
        m1, m0 = w1.loc[w1.has_q2 == 1, y].mean(), w0.loc[w0.has_q2 == 1, y].mean()
        num += len(g) * (p1 * m1 - p0 * m0); den += len(g) * (p1 - p0)
    return num / den, den / len(v)


q3 = []
for k in ['SP_S1', 'SP_S0', 'UD_A']:
    v = S[k][S[k].ok90].copy()
    for y in ['edge_share', 'ret_turn', 'win_leg']:
        est, share = complier(v, y)
        bs = np.array([complier(v.iloc[ix], y) for ix in boot_idx(v, CL[k[:2]], 300, SEED + 7)])
        q3.append(dict(sample=k, metric=y, complier_mean=est, ci95_lo=np.percentile(bs[:, 0], 2.5), ci95_hi=np.percentile(bs[:, 0], 97.5),
                       complier_share=share, share_ci95_lo=np.percentile(bs[:, 1], 2.5), share_ci95_hi=np.percentile(bs[:, 1], 97.5),
                       share_t=share / bs[:, 1].std(),
                       assumptions='monotone selection within price bins; a win does not change the always-traders mean; '
                                   'win as good as random within price bins'))
q3 = pd.DataFrame(q3)
q3.to_csv(f'{R2RES}/T4_q3.csv', index=False)
print(q3.round(4).to_string())
log('done')
