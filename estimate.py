"""Re-estimate the current crypto A and sports S1 effects.

Every displayed interval is a 95% CI from SEs clustered two-way by market (crypto) or game (sports) and by the UTC day
of the first market's resolution, on the same model and sample as its point estimate.

Parts
  M   own-window one-week (and 1/30/90-day) effects, samples A and S1                       -> headline effects
  C   cumulative return on a dense horizon grid (15 min to 90 days), common 90-day sample       -> return curves
      standardized levels: lost = mean_i yhat_i(win=0), won = lost + beta (linear g-computation, same model)
  P   any activity in disjoint periods after resolution, common 90-day sample
  D   B1: cumulative 7-day versus 30/90-day effect, estimated as the effect on (ret_h - ret7d), common sample
  R   robustness of the one-week effect: market FE, rich pre-order controls, timing, schedule flags,
      lead >= 1 h; bootstrap intervals and placebo / simulated-null p values are read from data/checks
  E   early / late period effects (own window; pooled model, period-specific FE and controls)
  K   calibration: win rate by 5-cent price bin
  Q   later bets among returners, read from data/checks
Outputs: results/ledger.csv and results/curves.csv.
"""
import os
from common import *               # noqa: E402  (WORK, RES, DAY, log)
import numpy as np, pandas as pd   # noqa: E402
from scipy import stats            # noqa: E402
from multiprocessing import Pool   # noqa: E402
from samples import load_sample, samples as samples_r2, spec   # noqa: E402
from ols import build, ols_fit, cluster_V       # noqa: E402

OUT = RES
os.makedirs(OUT, exist_ok=True)
CHECKS = f'{WORK}/checks'
VERSION = 'crypto A and sports S1'
CL = {'UD': 'cid', 'SP': 'event'}
DOM = {'UD': 'crypto', 'SP': 'sports'}
SAMP = {'UD_A': 'A', 'SP_S1': 'S1'}

d = load_sample()
S = samples_r2(d)
for k in SAMP:
    S[k] = S[k].copy()


def vcov2(X, e, XtX_inv, g1, g2):
    V1, _ = cluster_V(X, e, XtX_inv, g1)
    V2, _ = cluster_V(X, e, XtX_inv, g2)
    g12 = pd.factorize(pd.Series(g1).astype(str) + '|' + pd.Series(g2).astype(str))[0]
    V12, _ = cluster_V(X, e, XtX_inv, g12)
    return V1, V1 + V2 - V12


def fit(s, y, ctrl, cat, clcol, x='win', inter=None, fe=None):
    """OLS with dummy FE; optional absorbed FE (within transformation).  Returns coefficients and one-/two-way V."""
    cols = list(dict.fromkeys([y, x, clcol, 'rday', 'cid', 'event'] + list(ctrl) + list(cat) + ([inter] if inter else []) + ([fe] if fe else [])))
    s = s[cols].dropna().reset_index(drop=True)
    if fe:
        s = s[s.groupby(fe)[x].transform('nunique') > 1].reset_index(drop=True)
    X, Y, names = build(s, y, x, list(ctrl), list(cat), inter=inter)
    if fe:
        gid = pd.factorize(s[fe])[0]
        X = X[:, :-1]; names = names[:-1]
        cnt = np.bincount(gid)
        X = X - (np.add.reduceat(X[np.argsort(gid, kind='stable')], np.r_[0, np.cumsum(cnt)[:-1]]) / cnt[:, None])[gid]
        Y = Y - (np.bincount(gid, Y) / cnt)[gid]
        keep = X.std(axis=0) > 1e-12
        X = X[:, keep]; names = [n for n, k in zip(names, keep) if k]
    beta, XtX_inv, e = ols_fit(X, Y)
    V1, V2 = vcov2(X, e, XtX_inv, s[clcol].to_numpy(), s.rday.to_numpy())
    return dict(beta=beta, V1=V1, V2=V2, names=names, s=s, Y=Y)


def summarize(f, j=0, scale=100, ycol=None, **meta):
    b, se1, se2 = f['beta'][j], np.sqrt(max(f['V1'][j, j], 0)), np.sqrt(max(f['V2'][j, j], 0))
    s = f['s']
    lost = s.win.to_numpy() == 0
    raw_y = s[ycol].to_numpy(float) if ycol in s else f['Y']
    return dict(estimate_pp=scale * b, se_1way=scale * se1, se_2way=scale * se2, ci_low_pp=scale * (b - 1.96 * se2),
                ci_high_pp=scale * (b + 1.96 * se2), p_2way=2 * stats.norm.sf(abs(b / se2)) if se2 > 0 else np.nan,
                n_wallets=len(s), n_markets=s.cid.nunique(), n_games=s.event.nunique(), n_rdays=s.rday.nunique(),
                base_lost_raw=scale * raw_y[lost].mean(), win_share=s.win.mean(), **meta)


def main_spec(k):
    sp = spec(k)
    return list(sp['ctrl']), list(sp['cat']), CL[k[:2]]


ROWS = []


def add(result_id, k, sample_rule, obs, outcome, horizon, est_type, model_spec, r, ci_method='two-way cluster', **extra):
    ROWS.append(dict(result_id=result_id, domain=DOM[k[:2]], sample_rule=sample_rule, observation_sample=obs, outcome=outcome,
                     horizon_or_period=horizon, estimate_type=est_type, ci_level=0.95, ci_method=ci_method,
                     cluster_dimensions=f'{CL[k[:2]]} x resolution day' if ci_method == 'two-way cluster' else '',
                     model_spec=model_spec, source_version=VERSION, **r, **extra))


MAINSPEC = {'UD': 'price-bin FE + month x window x side FE + hour FE + log stake + log time to scheduled window end',
            'SP': 'price-bin FE + month FE + hour FE + log stake'}

# ---- M: headline own-window effects ---------------------------------------------------------------------------------------
for k in SAMP:
    ctrl, cat, cl = main_spec(k)
    for y, h in [('ret1d', '1 day'), ('ret7d', '7 days'), ('ret30d', '30 days'), ('ret90d', '90 days')]:
        f = fit(S[k], y, ctrl, cat, cl)
        add(f'M:{k}:{y}', k, SAMP[k], 'own window', 'traded again (any fill)', h, 'regression coefficient', MAINSPEC[k[:2]],
            summarize(f, ycol=y))
log('M done')


# ---- C: dense cumulative-return grid on the common 90-day sample ---------------------------------------------------------
GRID_H = sorted(set(np.round(np.geomspace(0.25, 90 * 24, 34), 4).tolist() + [24, 7 * 24, 14 * 24, 30 * 24, 60 * 24, 90 * 24]))


def curve_job(args):
    k, hh = args
    ctrl, cat, cl = main_spec(k)
    v = S[k][S[k].ok90].copy()
    v['yh'] = (v.t_next.fillna(np.inf) <= hh * 3600).astype(float)
    f = fit(v, 'yh', ctrl, cat, cl)
    b = f['beta'][0]; se2 = np.sqrt(f['V2'][0, 0]); se1 = np.sqrt(f['V1'][0, 0])
    Y = f['Y']; w = f['s'].win.to_numpy(float)
    lost_std = Y.mean() - b * w.mean()
    return dict(sample=k, hours=hh, days=hh / 24, b=100 * b, se_1way=100 * se1, se_2way=100 * se2, lo=100 * (b - 1.96 * se2),
                hi=100 * (b + 1.96 * se2), lost_std=100 * lost_std, won_std=100 * (lost_std + b),
                lost_raw=100 * Y[w == 0].mean(), won_raw=100 * Y[w == 1].mean(), n=len(Y))


# ---- P / D: periods and horizon differences on the common sample -----------------------------------------------------------
PERIODS = [('act0_7', '1-7'), ('act8_14', '8-14'), ('act15_21', '15-21'), ('act22_30', '22-30'), ('act31_60', '31-60'), ('act61_90', '61-90')]

if __name__ == '__main__':
    with Pool(min(48, os.cpu_count() or 4)) as pool:
        cur = pd.DataFrame(pool.map(curve_job, [(k, h) for k in SAMP for h in GRID_H]))
    cur.to_csv(f'{OUT}/curves.csv', index=False)
    for _, r in cur[cur.hours.isin([24, 168, 720, 1440, 2160])].iterrows():
        ROWS.append(dict(result_id=f'C:{r["sample"]}:{r.hours:g}h', domain=DOM[r['sample'][:2]], sample_rule=SAMP[r['sample']],
                         observation_sample='common 90d', outcome='traded again (any fill)', horizon_or_period=f'{r.days:g} days',
                         estimate_type='regression coefficient', estimate_pp=r.b, se_1way=r.se_1way, se_2way=r.se_2way,
                         ci_low_pp=r.lo, ci_high_pp=r.hi, ci_level=0.95, ci_method='two-way cluster', n_wallets=r.n,
                         lost_std_pp=r.lost_std, won_std_pp=r.won_std, base_lost_raw=r.lost_raw,
                         cluster_dimensions=f'{CL[r["sample"][:2]]} x resolution day', model_spec=MAINSPEC[r['sample'][:2]],
                         source_version=VERSION))
    log('C done')

    for k in SAMP:
        ctrl, cat, cl = main_spec(k)
        v = S[k][S[k].ok90].copy()
        for y, lab in PERIODS:
            f = fit(v, y, ctrl, cat, cl)
            add(f'P:{k}:{y}', k, SAMP[k], 'common 90d', 'any activity in period', f'days {lab}', 'regression coefficient',
                MAINSPEC[k[:2]], summarize(f, ycol=y))
        for y, lab in [('days30', 'active days, days 1-30'), ('weeks90', 'active weeks, weeks 1-13')]:
            f = fit(v, y, ctrl, cat, cl)
            add(f'P:{k}:{y}', k, SAMP[k], 'common 90d', lab, '', 'regression coefficient (count)', MAINSPEC[k[:2]],
                summarize(f, scale=1, ycol=y))
        for h in ['ret30d', 'ret90d']:
            v[f'd_{h}'] = v[h] - v.ret7d
            f = fit(v, f'd_{h}', ctrl, cat, cl)
            add(f'D:{k}:{h}-ret7d', k, SAMP[k], 'common 90d', f'{h} minus ret7d (B1)', f'{h[3:]} vs 7d', 'difference of coefficients',
                MAINSPEC[k[:2]], summarize(f, ycol=f'd_{h}'))
    log('P/D done')

    # ---- R: robustness of the one-week effect, own window ------------------------------------------------------------------
    for k in SAMP:
        ctrl, cat, cl = main_spec(k)
        f = fit(S[k], 'ret7d', ctrl, ['pbin'], cl, fe='cid')
        add(f'R:{k}:market_fe', k, SAMP[k], 'own window, markets with both sides', 'traded again', '7 days', 'regression coefficient',
            'market FE + price-bin FE + main controls', summarize(f, ycol='ret7d'))
    ud = pd.read_parquet(f'{WORK}/crypto_features.parquet')
    ud['lsched'] = np.log1p(ud.tend - ud.ts0)
    ud['rday'] = (ud.res_ts // DAY).astype(int)
    for c in ['r15m', 'r1h', 'r4h', 'r24h', 'vol24', 'prev1', 'prev4']:
        ud[c] = ud[c].fillna(ud[c].median())
    ud['asset_hz'] = ud.asset + '_' + ud.hz
    RICH = ['r15m', 'r1h', 'r4h', 'r24h', 'vol24', 'prev1', 'prev4', 'lfills0', 'llead']
    ctrl, cat, cl = main_spec('UD_A')
    f = fit(ud, 'ret7d', ctrl, cat, cl)
    add('R:UD_A:rich_same_rows_main', 'UD_A', 'A', 'own window, rows with features', 'traded again', '7 days', 'regression coefficient',
        MAINSPEC['UD'], summarize(f, ycol='ret7d'))
    f = fit(ud, 'ret7d', ctrl + RICH, cat + ['asset_hz', 'dow'], cl)
    add('R:UD_A:rich', 'UD_A', 'A', 'own window, rows with features', 'traded again', '7 days', 'regression coefficient',
        'main + prior Binance returns (15m/1h/4h/24h), 24h volatility, previous 1/4 window outcomes, fills, lead, asset x window, day of week',
        summarize(f, ycol='ret7d'))
    v = S['SP_S1'].copy()
    v['lead1h'] = (v.ind_start - v.ts0) >= 3600
    ctrl, cat, cl = main_spec('SP_S1')
    f = fit(v[v.lead1h], 'ret7d', ctrl, cat, cl)
    add('R:SP_S1:lead1h', 'SP_S1', 'S1, order >= 1 h before scheduled start', 'own window', 'traded again', '7 days',
        'regression coefficient', MAINSPEC['SP'], summarize(f, ycol='ret7d'))
    # Saved checks. These rows are not re-estimated here.
    a2 = pd.read_csv(f'{CHECKS}/A2_time_controls.csv')
    lab = {('UD', 'M0'): 'actual time to resolution', ('UD', 'M2'): 'no timing term', ('UD', 'M1'): 'scheduled window end (main)',
           ('SPM', 'M1'): 'bins of time to scheduled start', ('SPR', 'M1'): 'rich controls (league, lead bins, type, side, fills, 1c bins, dow)'}
    for (key, m), txt in lab.items():
        r = a2[(a2.key == key) & (a2.spec == m) & (a2.y == 'ret7d')].iloc[0]
        k = 'UD_A' if key == 'UD' else 'SP_S1'
        add(f'R:{k}:A2_{key}_{m}', k, SAMP[k], 'own window', 'traded again', '7 days', 'regression coefficient', txt,
            dict(estimate_pp=r.b, se_1way=r.se_cluster, se_2way=r.se_2way_rday, ci_low_pp=r.b - 1.96 * r.se_2way_rday,
                 ci_high_pp=r.b + 1.96 * r.se_2way_rday, n_wallets=r.n, boot_rday_lo=r.boot_lo, boot_rday_hi=r.boot_hi))
    a1 = pd.read_csv(f'{CHECKS}/A1_sensitivity.csv')
    r = a1[a1['sample'].str.startswith('excl. F1-F3 (') & (a1.y == 'ret7d')].iloc[0]
    add('R:SP_S1:excl_nba_flags', 'SP_S1', 'S1 excluding flagged NBA orders (0.37%)', 'own window', 'traded again', '7 days',
        'regression coefficient', MAINSPEC['SP'], dict(estimate_pp=r.b, se_1way=r.se_game, se_2way=r.se_2way_rday,
                                                       ci_low_pp=r.lo_2way, ci_high_pp=r.hi_2way, n_wallets=r.n, n_games=r.games))
    t5 = pd.read_csv(f'{CHECKS}/T5_persistent_activity.csv')
    t6 = pd.read_csv(f'{CHECKS}/T6_null_model_sensitivity.csv')
    for k in SAMP:
        r = t5[(t5['sample'] == k) & (t5.y == 'ret7d') & (t5.frame == 'own window')].iloc[0]
        for blk in ['rday', 'rweek']:
            add(f'R:{k}:boot_{blk}', k, SAMP[k], 'own window', 'traded again', '7 days', 'regression coefficient', MAINSPEC[k[:2]],
                dict(estimate_pp=100 * r.b, ci_low_pp=100 * r[f'boot_{blk}_lo'], ci_high_pp=100 * r[f'boot_{blk}_hi'], n_wallets=r.n),
                ci_method=f'block bootstrap of resolution {blk[1:]}s (999)')
    r = t6[(t6.test == 'up/down time-shift placebo') & (t6['sample'] == 'UD_A') & (t6.y == 'ret7d') & (t6.version == 'common support')].iloc[0]
    add('R:UD_A:placebo', 'UD_A', 'A, common support of 180 time shifts', 'own window', 'traded again', '7 days', 'rank p',
        '', dict(estimate_pp=100 * r.real, p_2way=r.p, n_wallets=r.n), ci_method='time-shift placebo rank')
    r = t6[t6.test.str.startswith('sports') & (t6['sample'] == 'SP_S1') & (t6.y == 'ret7d') & (t6.version == 'one market per game') &
           (t6.prob_model == 'cal_time_draw')].iloc[0]
    add('R:SP_S1:simnull', 'SP_S1', 'S1, one market per game', 'own window', 'traded again', '7 days', 'simulated-null p', '',
        dict(estimate_pp=100 * r.real, p_2way=r.p, n_wallets=r.n), ci_method='price-model simulated null (2,000 draws)')
    log('R done')

    # ---- E: early / late periods, pooled model with period-specific FE and controls ---------------------------------------
    for k in SAMP:
        ctrl, cat, cl = main_spec(k)
        v = S[k].copy()
        for c in cat:
            v[c] = v[c].astype(str) + '_' + v.late.astype(int).astype(str)
        ctl = list(ctrl)
        for c in ctrl:
            v[f'{c}_late'] = v[c] * v.late; ctl.append(f'{c}_late')
        for y in ['ret7d', 'act22_30', 'act61_90']:
            f = fit(v, y, ctl, cat, cl, inter='late')
            V = f['V2']; b = f['beta']
            late_b, late_se = b[0] + b[1], np.sqrt(V[0, 0] + V[1, 1] + 2 * V[0, 1])
            s = f['s']
            for per, est, se, n in [('early', b[0], np.sqrt(V[0, 0]), int((s.late == 0).sum())), ('late', late_b, late_se, int((s.late == 1).sum())),
                                    ('late minus early', b[1], np.sqrt(V[1, 1]), len(s))]:
                add(f'E:{k}:{y}:{per}', k, SAMP[k], 'own window', y, per, 'regression coefficient', 'pooled, period-specific FE and controls',
                    dict(estimate_pp=100 * est, se_2way=100 * se, ci_low_pp=100 * (est - 1.96 * se), ci_high_pp=100 * (est + 1.96 * se),
                         p_2way=2 * stats.norm.sf(abs(est / se)), n_wallets=n,
                         period_split='2026-01-01' if k == 'UD_A' else '2025-07-01'))
    log('E done')

    # ---- K: calibration by price bin; descriptives -------------------------------------------------------------------------
    import statsmodels.api as sm
    for k in SAMP:
        v = S[k]
        for pb, g in v.groupby('pbin'):
            yv = g.win.to_numpy(float); m = yv.mean()
            gs = np.bincount(pd.factorize(g[CL[k[:2]]])[0], yv - m); G = len(gs)
            se = np.sqrt(G / max(G - 1, 1) * (gs ** 2).sum()) / len(yv)
            ROWS.append(dict(result_id=f'K:{k}:pbin{pb}', domain=DOM[k[:2]], sample_rule=SAMP[k], observation_sample='all',
                             outcome='first side won', horizon_or_period=f'price {pb * 5}-{pb * 5 + 5} c', estimate_type='mean',
                             estimate_pp=100 * m, ci_low_pp=100 * (m - 1.96 * se), ci_high_pp=100 * (m + 1.96 * se), ci_level=0.95,
                             ci_method=f'cluster by {CL[k[:2]]}', n_wallets=len(g), mean_price_c=100 * g.p0.mean(),
                             source_version=VERSION))
        lead = (v.tstart - v.ts0) / 60 if k == 'UD_A' else (v.ind_start - v.ts0) / 3600
        ROWS.append(dict(result_id=f'S:{k}', domain=DOM[k[:2]], sample_rule=SAMP[k], observation_sample='all', estimate_type='descriptive',
                         n_wallets=len(v), n_markets=v.cid.nunique(), n_games=v.event.nunique(), win_share=v.win.mean(),
                         median_stake=v.stake0.median(), median_lead=lead.median(), lead_unit='minutes' if k == 'UD_A' else 'hours',
                         median_order_to_resolution_h=((v.res_ts - v.ts0) / 3600).median(), n_common90=int(v.ok90.sum()),
                         first_order=str(pd.to_datetime(v.ts0.min(), unit='s').date()), last_order=str(pd.to_datetime(v.ts0.max(), unit='s').date()),
                         never_return_share=v.t_next.isna().mean(), source_version=VERSION))

    # ---- Q: later bets among returners (saved comparison) --------------------------------------------------------------
    q = pd.read_csv(f'{CHECKS}/T4_quality_and_equivalence.csv')
    cov = pd.read_csv(f'{CHECKS}/T4_coverage.csv').set_index('sample')
    for k in SAMP:
        for met, mg, sc in [('edge_share', 0.01, 100), ('ret_turn', 0.02, 100), ('win_leg', 0.02, 100)]:
            r = q[(q['sample'] == k) & (q.metric == met) & (q.weight == 'wallet-equal (main)') & np.isclose(q.margin, mg)].iloc[0]
            ROWS.append(dict(result_id=f'Q:{k}:{met}', domain=DOM[k[:2]], sample_rule=SAMP[k], observation_sample='common 90d, returners with later trades',
                             outcome=met, estimate_type='conditional difference (descriptive)', estimate_pp=sc * r.b, se_1way=sc * r.se,
                             ci_low_pp=sc * r.ci90_lo, ci_high_pp=sc * r.ci90_hi, ci_level=0.90, ci_method=f'cluster by {r.cluster}',
                             p_2way=r.p_tost, margin=sc * mg, n_wallets=r.n, base_lost_raw=sc * r.base,
                             share_trade_win=cov.loc[k, 'share_trade_win'], share_trade_lose=cov.loc[k, 'share_trade_lose'],
                             source_version='saved later-trade comparison'))

    led = pd.DataFrame(ROWS)
    led.to_csv(f'{OUT}/ledger.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_colwidth', 40)
    show = ['result_id', 'observation_sample', 'horizon_or_period', 'estimate_pp', 'se_1way', 'se_2way', 'ci_low_pp', 'ci_high_pp', 'p_2way', 'n_wallets', 'base_lost_raw']
    print(led[[c for c in show if c in led]].round(3).to_string())
    print(cur.round(3).to_string())
    log('done')
