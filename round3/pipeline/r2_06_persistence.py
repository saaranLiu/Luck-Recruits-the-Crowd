"""T5: persistent activity.  Robust inference for returning and later activity, a direct Win x period test, and the
sources of the round-1 text/table differences.

Frames: 'common 90d' = newcomers whose 90-day window is complete (one sample for every outcome; main dynamic frame);
        'own window' = newcomers whose window for that outcome is complete (supplement).
Outcomes (time origin = first market's resolution; windows left-open, right-closed in seconds):
        ret7d = any fill in (0, 7d]; act22_30 = any fill in (21d, 30d]; act61_90 = any fill in (60d, 90d];
        ret30d, days30 (distinct days with a fill, day index 0-30), weeks90 (distinct weeks, index 0-12).
Inference, all reported: (a) clustered by market (up/down) or game (sports); (b) two-way: market/game x resolution day
        (main, fixed before estimation); (c) two-way: market/game x order day; (d) bootstrap of resolution days;
        (e) bootstrap of resolution weeks (longer blocks for serial dependence).  999 draws.
Period test: pooled regression with Win x late, all fixed effects and controls interacted with the period.
"""
from r2_common import *
import numpy as np, pandas as pd
from multiprocessing import Pool
from r2_samples import load_r2, samples_r2, spec
from r2_fast import build, ols_fit, cluster_se, block_boot

B = 999
d = load_r2()
S = samples_r2(d)
KEYS = ['UD_A', 'SP_S1', 'SP_S0']
YS = ['ret7d', 'act22_30', 'act61_90', 'ret30d', 'days30', 'weeks90']
CL = {'UD': 'cid', 'SP': 'event'}


def frame(k, y, fr):
    v = S[k]
    if fr == 'common 90d':
        v = v[v.ok90]
    sp = spec(k)
    cols = list(dict.fromkeys([y, 'win', 'cid', 'event', 'rday', 'oday', 'rweek', 'late', 'ts0'] + list(sp['ctrl']) + list(sp['cat'])))
    return v[cols].dropna(subset=[y] + list(sp['ctrl'])).reset_index(drop=True), sp


def job(args):
    k, y, fr = args
    s, sp = frame(k, y, fr)
    X, Y, _ = build(s, y, 'win', sp['ctrl'], sp['cat'])
    beta, XtX_inv, e = ols_fit(X, Y)
    g = s[CL[k[:2]]].to_numpy()
    rng = np.random.default_rng(SEED + len(s))
    out = dict(sample=k, y=y, frame=fr, b=beta[0], n=len(s), n_clusters=len(np.unique(g)), n_rdays=s.rday.nunique(),
               base=Y[s.win.to_numpy() == 0].mean(), first=pd.to_datetime(s.ts0.min(), unit='s').date(),
               last=pd.to_datetime(s.ts0.max(), unit='s').date())
    out['se_cluster'], _ = cluster_se(X, e, XtX_inv, g)
    out['se_2way_rday'], _ = cluster_se(X, e, XtX_inv, [g, s.rday.to_numpy()])
    out['se_2way_oday'], _ = cluster_se(X, e, XtX_inv, [g, s.oday.to_numpy()])
    for lab, blk in [('boot_rday', 'rday'), ('boot_rweek', 'rweek')]:
        bs = block_boot(X, Y, s[blk].to_numpy(), B, rng)
        out[f'se_{lab}'] = bs.std(); out[f'{lab}_lo'] = np.percentile(bs, 2.5); out[f'{lab}_hi'] = np.percentile(bs, 97.5)
    out['main_lo'] = out['b'] - 1.96 * out['se_2way_rday']; out['main_hi'] = out['b'] + 1.96 * out['se_2way_rday']
    out['main_p'] = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(out['b'] / out['se_2way_rday'])))
    out['max_se'] = max(out['se_cluster'], out['se_2way_rday'], out['se_2way_oday'], out['se_boot_rday'], out['se_boot_rweek'])
    return out


def period_job(args):
    k, y, fr = args
    s, sp = frame(k, y, fr)
    s = s.copy()
    for c in sp['cat']:
        s[c] = s[c].astype(str) + '_' + s.late.astype(int).astype(str)
    ctrl = list(sp['ctrl'])
    for c in sp['ctrl']:
        s[f'{c}_late'] = s[c] * s.late; ctrl.append(f'{c}_late')
    s['late_f'] = s.late
    X, Y, names = build(s, y, 'win', ctrl, list(sp['cat']), inter='late')
    beta, XtX_inv, e = ols_fit(X, Y)
    g = s[CL[k[:2]]].to_numpy()
    se_c, _ = cluster_se(X, e, XtX_inv, g, j=1)
    se_2, _ = cluster_se(X, e, XtX_inv, [g, s.rday.to_numpy()], j=1)
    bs = block_boot(X, Y, s.rday.to_numpy(), B, np.random.default_rng(SEED + 99), j=1)
    from scipy import stats
    early = s[s.late == 0]; late = s[s.late == 1]
    return dict(sample=k, y=y, frame=fr, b_early=beta[0], b_diff_late_minus_early=beta[1], se_cluster=se_c, se_2way_rday=se_2,
                se_boot_rday=bs.std(), boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5),
                p_2way=2 * (1 - stats.norm.cdf(abs(beta[1] / se_2))), n_early=len(early), n_late=len(late),
                late_first=pd.to_datetime(late.ts0.min(), unit='s').date() if len(late) else None,
                late_last=pd.to_datetime(late.ts0.max(), unit='s').date() if len(late) else None)


if __name__ == '__main__':
    jobs = [(k, y, fr) for k in KEYS for y in YS for fr in ['common 90d', 'own window']]
    with Pool(40) as pool:
        res = pd.DataFrame(pool.map(job, jobs))
    res.to_csv(f'{R2RES}/T5_persistent_activity.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
    print(res[['sample', 'y', 'frame', 'b', 'se_cluster', 'se_2way_rday', 'se_2way_oday', 'se_boot_rday', 'se_boot_rweek', 'main_lo', 'main_hi', 'n', 'first', 'last']].round(4).to_string())
    pj = [(k, y, 'own window') for k in KEYS for y in ['ret7d', 'act22_30', 'act61_90', 'ret30d']]
    with Pool(16) as pool:
        per = pd.DataFrame(pool.map(period_job, pj))
    per.to_csv(f'{R2RES}/T5_period_interaction.csv', index=False)
    print(per.round(4).to_string())
    # sources of the round-1 differences: same outcome, two frames, and the newcomers only in the own-window frame
    rows = []
    for k, y in [('SP_S0', 'act22_30'), ('UD_A', 'ret30d'), ('SP_S1', 'act22_30')]:
        v = S[k]
        own = v[v[y].notna()]; com = own[own.ok90]; extra = own[~own.ok90]
        for lab, s in [('own window', own), ('common 90d', com), ('own window only (not in common 90d)', extra)]:
            sp = spec(k)
            ss = s.dropna(subset=[y] + list(sp['ctrl']))
            X, Y, _ = build(ss, y, 'win', sp['ctrl'], sp['cat'])
            beta, XtX_inv, e = ols_fit(X, Y)
            se, _ = cluster_se(X, e, XtX_inv, ss[CL[k[:2]]].to_numpy())
            rows.append(dict(sample=k, y=y, frame=lab, b=beta[0], se=se, n=len(ss), first=pd.to_datetime(ss.ts0.min(), unit='s').date(),
                             last=pd.to_datetime(ss.ts0.max(), unit='s').date()))
    pd.DataFrame(rows).to_csv(f'{R2RES}/T5_frame_differences.csv', index=False)
    print(pd.DataFrame(rows).round(4).to_string())
    log('done')
