"""T6: sensitivity of the design-based nulls.

Sports: a price-model-based simulated null.  Each market's reference leg is redrawn from a probability model of the
outcome; all wallets in the market share the draw, so opposite sides always get opposite outcomes.  The real
estimate is compared with the null distribution of the same regression.  Probability models:
  base        q_m = mean probability implied by the sample's first orders in market m (round 1);
  cal_time    logistic recalibration logit P(Y=1) = a + b logit(q_m) + c 1[q_m>0.5], fitted out of fold with
              calendar-month folds (each month predicted from the other months);
  cal_game    same model, 10 folds grouped by game;
  cal_time_draw  cal_time with parameter uncertainty: each null draw uses (a,b,c) drawn from N(beta_hat, V_hat) of
              the fold's fit.
Versions: all markets; one market per game (the market with most sample wallets).  2,000 draws; p = (k+1)/(N+1),
k = #draws with |null| >= |real|.  Outcomes ret7d, ret30d (own window), act61_90 (common 90d frame).
Up/down time shift: the win is recomputed from the realised outcome of the same asset and window length shifted by
s x 4h (s = +-1..+-90).  (a) round-1 version: each shift uses all wallets with a defined shifted outcome;
(b) common support: only wallets with a defined outcome at every shift, so all shifts use the same rows.
Reported: per-shift sample size, rank of the real estimate, p = (k+1)/(N+1).
"""
from r2_common import *
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.special import logit, expit
from r2_samples import load_r2, samples_r2, spec, ols2
from prep import design

R = 2000
rng = np.random.default_rng(SEED)
d = load_r2()
S = samples_r2(d)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)


def fixed_design(s, sp):
    X = np.column_stack([np.ones(len(s)), design(s, 'win', sp['ctrl'], sp['cat']).drop(columns='win').to_numpy(float)])
    Q, _ = np.linalg.qr(X)
    return Q


def calib_feats(q):
    q = np.clip(q, 1e-3, 1 - 1e-3)
    return np.column_stack([np.ones(len(q)), logit(q), (q > 0.5).astype(float)])


def crossfit(mk, folds, draws=0):
    """Out-of-fold calibrated probability for each market; with draws>0 returns (n_mkt, draws) from N(beta, V)."""
    out = np.empty(len(mk)) if draws == 0 else np.empty((len(mk), draws))
    coefs = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = sm.GLM(mk.y_ref[tr], calib_feats(mk.q[tr].to_numpy()), family=sm.families.Binomial()).fit()
        Xte = calib_feats(mk.q[te].to_numpy())
        coefs.append(m.params)
        if draws == 0:
            out[te] = expit(Xte @ m.params)
        else:
            bd = rng.multivariate_normal(m.params, m.cov_params(), size=draws)
            out[te] = expit(Xte @ bd.T)
    return out, np.array(coefs)


def scores(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(brier=float(np.mean((p - y) ** 2)), logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                mean_p=float(p.mean()), mean_y=float(y.mean()))


ri, cal = [], []
for k in ['SP_S1', 'SP_S0']:
    base = S[k].copy()
    base['side'] = base.D0 == 1
    base['y_ref'] = np.where(base.side, base.win, 1 - base.win)
    base['q_i'] = np.where(base.side, base.p0, 1 - base.p0)
    mk = base.groupby('cid').agg(q=('q_i', 'mean'), y_ref=('y_ref', 'first'), event=('event', 'first'), month=('month', 'first'),
                                 nw=('w', 'size')).reset_index()
    mk['tfold'] = pd.factorize(mk.month)[0]
    ev = mk.event.unique(); ef = dict(zip(ev, rng.permutation(len(ev)) % 10))
    mk['gfold'] = mk.event.map(ef)
    probs = {'base': mk.q.to_numpy()}
    probs['cal_time'], ct = crossfit(mk, mk.tfold.to_numpy())
    probs['cal_game'], cg = crossfit(mk, mk.gfold.to_numpy())
    pdraw, _ = crossfit(mk, mk.tfold.to_numpy(), draws=R)
    full = sm.GLM(mk.y_ref, calib_feats(mk.q.to_numpy()), family=sm.families.Binomial()).fit()
    for nm, p in probs.items():
        cal.append(dict(sample=k, model=nm, n_mkt=len(mk), **scores(p, mk.y_ref.to_numpy())))
    cal.append(dict(sample=k, model='cal_time_draw (mean over draws)', n_mkt=len(mk), **scores(pdraw.mean(1), mk.y_ref.to_numpy())))
    cal.append(dict(sample=k, model='full-sample fit: a, b, c', a=full.params[0], b=full.params[1], c=full.params[2],
                    se_a=full.bse[0], se_b=full.bse[1], se_c=full.bse[2], fold_b_min=ct[:, 1].min(), fold_b_max=ct[:, 1].max()))
    mk_idx = dict(zip(mk.cid, range(len(mk))))
    top = base.groupby('event').cid.agg(lambda s: s.value_counts().index[0])
    sp = spec(k)
    for ver, v in [('all markets', base), ('one market per game', base[base.cid.isin(top.values)])]:
        for y, fr in [('ret7d', 'own window'), ('ret30d', 'own window'), ('act61_90', 'common 90d')]:
            vv = v[v.ok90] if fr == 'common 90d' else v
            s = vv[list(dict.fromkeys([y, 'win', 'cid', 'side'] + list(sp['ctrl']) + list(sp['cat'])))].dropna().reset_index(drop=True)
            Q = fixed_design(s, sp)
            Y = s[y].to_numpy(float); ey = Y - Q @ (Q.T @ Y)
            w = s.win.to_numpy(float); wt = w - Q @ (Q.T @ w)
            real = float(wt @ ey / (wt @ wt))
            m = s.cid.map(mk_idx).to_numpy()
            side = s.side.to_numpy()
            for nm in ['base', 'cal_time', 'cal_game', 'cal_time_draw']:
                null = np.empty(R)
                for j in range(0, R, 200):
                    qq = pdraw[:, j:j + 200] if nm == 'cal_time_draw' else probs[nm][:, None]
                    ys = (rng.random(size=(len(mk), 200)) < qq).astype(float)
                    W = np.where(side[:, None], ys[m], 1 - ys[m])
                    Wt = W - Q @ (Q.T @ W)
                    null[j:j + 200] = (Wt * ey[:, None]).sum(0) / (Wt ** 2).sum(0)
                kk = int((np.abs(null) >= abs(real)).sum())
                qm = probs['cal_time'] if nm == 'cal_time_draw' else probs[nm]
                ri.append(dict(sample=k, version=ver, y=y, frame=fr, prob_model=nm, real=real, null_mean=null.mean(), null_sd=null.std(),
                               null_q025=np.percentile(null, 2.5), null_q975=np.percentile(null, 97.5), k=kk, p=(kk + 1) / (R + 1),
                               n=len(s), n_mkt=len(np.unique(m)), sim_win_rate=float(np.where(side, qm[m], 1 - qm[m]).mean()),
                               act_win_rate=w.mean()))
                log(k, ver, y, nm, 'real', round(real, 4), 'null mean', round(null.mean(), 4), 'sd', round(null.std(), 4), 'p', ri[-1]['p'])
ri = pd.DataFrame(ri); cal = pd.DataFrame(cal)
cal.to_csv(f'{R2RES}/T6_calibration.csv', index=False)
print(cal.round(4).to_string())

# ---- up/down time shift --------------------------------------------------------------------------------------------
us = pd.read_parquet(f'{WORK}/ud_series.parquet').set_index(['asset', 'hz', 'tstart']).up
SH = [s for s in range(-90, 91) if s != 0]
ts = []
for k in ['UD_A', 'UD_C']:
    v = S[k].copy().reset_index(drop=True)
    sp = spec(k)
    fake = {}
    for sh in SH:
        fu = us.reindex(list(zip(v.asset, v.hz, v.tstart + sh * 4 * 3600))).to_numpy()
        fake[sh] = np.where(~np.isnan(fu), (v.side_up == fu).astype(float), np.nan)
    F = pd.DataFrame(fake)
    common = F.notna().all(1).to_numpy()
    for y, fr in [('ret7d', 'own window'), ('ret30d', 'own window'), ('act61_90', 'common 90d')]:
        vv = v[v.ok90] if fr == 'common 90d' else v
        idx = vv.index
        # (a) per-shift available rows (round 1)
        realA = ols2(vv, y, **sp)
        nulls_a, ns_a = [], []
        for sh in SH:
            vv = vv.assign(fake=F.loc[idx, sh].to_numpy())
            r = ols2(vv, y, x='fake', **sp)
            nulls_a.append(r['b']); ns_a.append(r['n'])
        # (b) common support: same rows for real and every shift
        s = vv[common[idx]]
        s = s[list(dict.fromkeys([y, 'win', 'cid'] + list(sp['ctrl']) + list(sp['cat'])))].dropna()
        Q = fixed_design(s, sp)
        Y = s[y].to_numpy(float); ey = Y - Q @ (Q.T @ Y)
        w = s.win.to_numpy(float); wt = w - Q @ (Q.T @ w)
        realB = float(wt @ ey / (wt @ wt))
        Wm = F.loc[s.index, SH].to_numpy(float)
        Wt = Wm - Q @ (Q.T @ Wm)
        nulls_b = (Wt * ey[:, None]).sum(0) / (Wt ** 2).sum(0)
        for lab, real, nl, ns, n_real in [('per-shift rows (round 1)', realA['b'], np.array(nulls_a), np.array(ns_a), realA['n']),
                                          ('common support', realB, nulls_b, np.full(len(SH), len(s)), len(s))]:
            kk = int((np.abs(nl) >= abs(real)).sum())
            ts.append(dict(sample=k, y=y, frame=fr, version=lab, real=real, n_real=n_real, n_shift_min=int(ns.min()),
                           n_shift_max=int(ns.max()), n_shifts=len(nl), null_mean=nl.mean(), null_sd=nl.std(), k=kk,
                           p=(kk + 1) / (len(nl) + 1), rank_abs=int((np.abs(nl) > abs(real)).sum()) + 1,
                           note='shifts are dependent; p is a rank statistic, not an exact randomization p'))
        pd.DataFrame(dict(shift_h=np.array(SH) * 4, b=nulls_b)).assign(sample=k, y=y).to_csv(
            f'{R2WORK}/T6_shift_draws_{k}_{y}.csv', index=False)
        log(k, y, ts[-2], ts[-1])
ts = pd.DataFrame(ts)
ri['test'] = 'sports price-model-based simulated null'
ts['test'] = 'up/down time-shift placebo'
out = pd.concat([ri, ts], ignore_index=True)
out.to_csv(f'{R2RES}/T6_null_model_sensitivity.csv', index=False)
print(ri[['sample', 'version', 'y', 'prob_model', 'real', 'null_mean', 'null_sd', 'p', 'n', 'sim_win_rate', 'act_win_rate']].round(4).to_string())
print(ts.drop(columns=['note', 'test']).round(4).to_string())
log('done')
