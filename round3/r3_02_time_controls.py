"""A2: timing controls that use the actual on-chain resolution time versus scheduled (ex-ante) times.

Code audit (round 2):
  crypto main model   ldelay   = log1p(res_ts - ts0); res_ts is the on-chain ConditionResolution block time (ex post)
  sports rich model   lead_bin = bins of (res_ts - ts0) in hours, cut at [0, 6, 12, 24, 72, inf)   (ex post)
  sports main model   no timing term
Specifications, fixed before estimation; within each row only the timing term changes:
  M0  current implementation (actual time to resolution)
  M1  scheduled time: crypto log1p(tend - ts0), tend = window start + window length from the market slug;
      sports bins of (scheduled start - ts0) with the same cut points
  M2  no timing term
Every row uses the wallets for which M0, M1, and M2 are all defined (same N).  Inference as in round 2: SE clustered
by market (crypto) or game (sports), two-way with the UTC day of resolution, and 999 bootstraps of resolution days.
"""
from r3_common import *
import numpy as np, pandas as pd
from multiprocessing import Pool
from r2_samples import load_r2, samples_r2
from r2_fast import build, ols_fit, cluster_se, block_boot

B = 999
d = load_r2()
S = samples_r2(d)
CUTS = [0, 6, 12, 24, 72, np.inf]

ud = S['UD_A'].copy()
ud['t_m0'] = np.log1p(ud.res_ts - ud.ts0)
ud['t_m1'] = np.log1p(ud.tend - ud.ts0)
sp = S['SP_S1'].copy()
sp['lg'] = sp.league
sp['side_ref'] = (sp.D0 == 1).astype(float)
sp['pbin1'] = np.floor(sp.p0 * 100).astype(int)
sp['t_m0'] = pd.cut((sp.res_ts - sp.ts0) / 3600, CUTS, right=False).astype(str)
sp['t_m1'] = pd.cut((sp.ind_start - sp.ts0) / 3600, CUTS, right=False).astype(str)

MODELS = {
    'UD': dict(data=ud, ctrl=['lstake'], cat=['pbin', 'cell', 'hour'], cl='cid', kind='num',
               label='crypto A, main model'),
    'SPR': dict(data=sp, ctrl=['lstake', 'lfills0', 'side_ref'], cat=['pbin1', 'month', 'hour', 'lg', 'mtype', 'dow'], cl='event',
                kind='cat', label='sports S1, rich pre-bet controls'),
    'SPM': dict(data=sp, ctrl=['lstake'], cat=['pbin', 'month', 'hour'], cl='event', kind='cat',
                label='sports S1, main model (no timing term in the paper)'),
}


def job(args):
    key, y, m = args
    M = MODELS[key]
    s = M['data'].dropna(subset=[y, 't_m0', 't_m1'] + M['ctrl']).reset_index(drop=True)
    ctrl, cat = list(M['ctrl']), list(M['cat'])
    if m in ('M0', 'M1'):
        (ctrl if M['kind'] == 'num' else cat).append('t_' + m.lower())
    X, Y, _ = build(s, y, 'win', ctrl, cat)
    beta, XtX_inv, e = ols_fit(X, Y)
    g = s[M['cl']].to_numpy()
    se_c, _ = cluster_se(X, e, XtX_inv, g)
    se_2, _ = cluster_se(X, e, XtX_inv, [g, s.rday.to_numpy()])
    bs = block_boot(X, Y, s.rday.to_numpy(), B, np.random.default_rng(SEED + 3))
    return dict(model=M['label'], key=key, y=y, spec=m, n=len(s), clusters=len(np.unique(g)), b=100 * beta[0],
                se_cluster=100 * se_c, se_2way_rday=100 * se_2, boot_lo=100 * np.percentile(bs, 2.5), boot_hi=100 * np.percentile(bs, 97.5),
                lo_cluster=100 * (beta[0] - 1.96 * se_c), hi_cluster=100 * (beta[0] + 1.96 * se_c),
                base=100 * Y[s.win.to_numpy() == 0].mean())


if __name__ == '__main__':
    jobs = [(k, y, m) for k in MODELS for y in ['ret7d', 'ret30d'] for m in ['M0', 'M1', 'M2']]
    with Pool(18) as pool:
        res = pd.DataFrame(pool.map(job, jobs))
    res.to_csv(f'{R3RES}/A2_time_controls.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.drop(columns=['key']).round(3).to_string())

    # is the ex-post part of the timing related to the outcome?  winner-minus-loser within price bins and cells
    from r2_samples import ols2
    ud['lag_min'] = (ud.res_ts - ud.tend) / 60
    ud['lag_gt10'] = (ud.lag_min > 10).astype(float)
    sp['post_h'] = (sp.res_ts - sp.ind_start) / 3600
    sp['post_gt8'] = (sp.post_h > 8).astype(float)
    sp['lpost'] = np.log(sp.post_h.clip(lower=0.1))
    bal = []
    for v, ys, cat, cl in [(ud, ['lag_min', 'lag_gt10'], ('pbin', 'cell'), 'cid'), (sp, ['post_h', 'post_gt8', 'lpost'], ('pbin', 'month'), 'event')]:
        for y in ys:
            r = ols2(v, y, ctrl=(), cat=cat, cluster=cl)
            bal.append(dict(y=y, mean=v[y].mean(), median=v[y].median(), **{k: r[k] for k in ['b', 'se', 'p', 'n']}))
    bal = pd.DataFrame(bal)
    bal.to_csv(f'{R3RES}/A2_expost_timing_balance.csv', index=False)
    print(bal.round(4).to_string())
    log('done')
