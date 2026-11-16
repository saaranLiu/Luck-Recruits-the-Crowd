"""T1: sports time fields, independent schedule validation, and ex-ante sample sensitivity (S0-S3).

Independent schedules (downloaded from GitHub, not from Polymarket):
  NBA  sportsdataverse/hoopR-nba-data  nba_schedule_master.parquet   ESPN scheduled start (UTC)
  NHL  sportsdataverse/fastRhockey-nhl-data  nhl_schedule_master.parquet   NHL API scheduled start (UTC)
  NFL  nflverse/nfldata  games.csv   gameday + gametime (US Eastern)
These are final versions of the schedules (a rescheduled game shows its new time); no version as of the order exists.

Matching rule, fixed before any outcome is looked at: league, unordered pair of team codes after a fixed alias map,
and the slug date equal to the US-Eastern calendar date of the scheduled start.  A slug that matches no game or more
than one game is reported as unmatched or ambiguous.

Samples (all: first block in one market, aggressive, one side, price in [0.3, 0.7], resolved after the order):
  S0  current rule: order >= 6 h before the on-chain resolution (uses an ex-post time)
  S1  order before the independent scheduled start (NBA/NHL/NFL games with a unique match)
  S1p order before the metadata close time (all leagues; close time validated against S1 games below)
  S2  S1 and order >= 1 h before the independent scheduled start
  S3  S1 and the game was played as scheduled: resolution 1.5-8 h after the independent scheduled start
"""
from r2_common import *
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(R2))
from prep_ex import load_ex, samples_ex, ols2, spec

W = f'{WORK}/schedules'
ALIAS = {'nba': {'nyk': 'ny', 'gsw': 'gs', 'sas': 'sa', 'uta': 'utah', 'nop': 'no', 'was': 'wsh', 'pho': 'phx'},
         'nhl': {'mon': 'mtl', 'tb': 'tbl', 'las': 'vgk', 'utah': 'uta', 'nj': 'njd', 'cal': 'cgy', 'sj': 'sjs'},
         'nfl': {'lar': 'la', 'wsh': 'was', 'las': 'lv'}}

# ---- independent schedules ---------------------------------------------------------------------------------------
nba = pd.read_parquet(f'{W}/nba_schedule_master.parquet')
nba = pd.DataFrame({'lg': 'nba', 'a': nba.home_abbreviation.str.lower(), 'b': nba.away_abbreviation.str.lower(),
                    'start': pd.to_datetime(nba.date, utc=True), 'status': nba.status_type_name, 'gid': nba.game_id.astype(str)})
nhl = pd.read_parquet(f'{W}/nhl_schedule_master.parquet')
nhl = pd.DataFrame({'lg': 'nhl', 'a': nhl.home_team_abbr.str.lower(), 'b': nhl.away_team_abbr.str.lower(),
                    'start': pd.to_datetime(nhl.game_time, utc=True), 'status': nhl.game_state, 'gid': nhl.game_id.astype(str)})
nfl = pd.read_csv(f'{W}/nfl_games.csv')
nfl = nfl[nfl.gametime.notna()]
nfl = pd.DataFrame({'lg': 'nfl', 'a': nfl.home_team.str.lower(), 'b': nfl.away_team.str.lower(),
                    'start': pd.to_datetime(nfl.gameday + ' ' + nfl.gametime).dt.tz_localize('America/New_York').dt.tz_convert('UTC'),
                    'status': np.where(nfl.result.notna(), 'final', 'scheduled'), 'gid': nfl.game_id})
sch = pd.concat([nba, nhl, nfl], ignore_index=True)
sch = sch[sch.start >= '2022-10-01']
sch['date_et'] = sch.start.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
sch['pair'] = [tuple(sorted(x)) for x in zip(sch.a, sch.b)]
sch['start_ts'] = sch.start.astype('int64') // 10**9

# ---- Polymarket sports markets -----------------------------------------------------------------------------------
d = load_ex()
sp = d[d.grp == 'SportsML'].copy()
p = sp.event.str.extract(r'^([a-z]+)-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$')
sp['lg'], sp['t1'], sp['t2'], sp['sdate'] = p[0], p[1], p[2], p[3]
for lg, m in ALIAS.items():
    for c in ['t1', 't2']:
        sp.loc[sp.lg == lg, c] = sp.loc[sp.lg == lg, c].replace(m)
ev = sp.drop_duplicates('event')[['event', 'lg', 't1', 't2', 'sdate']].copy()
ev = ev[ev.lg.isin(['nba', 'nhl', 'nfl'])]
ev['pair'] = [tuple(sorted(x)) for x in zip(ev.t1, ev.t2)]
m = ev.merge(sch[['lg', 'pair', 'date_et', 'start_ts', 'status', 'gid']], left_on=['lg', 'pair', 'sdate'],
             right_on=['lg', 'pair', 'date_et'], how='left')
cnt = m.groupby('event').gid.transform('count')
m['match'] = np.where(cnt == 0, 'unmatched', np.where(cnt == 1, 'unique', 'ambiguous'))
m = m.drop_duplicates('event')
m.loc[m.match != 'unique', 'start_ts'] = np.nan
ev_match = m[['event', 'lg', 'match', 'start_ts', 'status', 'gid']].rename(columns={'start_ts': 'ind_start'})
ev_match.to_csv(f'{R2AUD}/time_validation_events.csv', index=False)
log('event match', ev_match.groupby(['lg', 'match']).size().to_dict())

sp = sp.merge(ev_match[['event', 'match', 'ind_start', 'status']], on='event', how='left')
sp['match'] = sp.match.fillna('not covered')
sp['ind_ok'] = sp.ind_start.notna()

# ---- proxy (metadata close time) against the independent start ---------------------------------------------------
base = sp.p0.between(0.3, 0.7)
mk = sp[sp.ind_ok].drop_duplicates('cid')
mk_err = (mk.close_ts - mk.ind_start) / 60
gap_res = (mk.res_ts - mk.ind_start) / 3600
proxy_rows = []
for lab, s in [('all matched markets', mk.index), ('moneyline', mk.index[mk.mtype == 'moneyline']),
               ('old validated proxy (res - close in [1.5h, 12h])', mk.index[mk.start_ok])]:
    e = mk_err.loc[s]
    proxy_rows.append(dict(set=lab, n_markets=len(e), median_min=e.median(), p05_min=e.quantile(0.05), p25_min=e.quantile(0.25),
                           p75_min=e.quantile(0.75), p95_min=e.quantile(0.95), share_abs_gt_60min=(e.abs() > 60).mean(),
                           share_abs_le_5min=(e.abs() <= 5).mean(), share_close_after_start=(e > 0).mean(),
                           res_minus_start_med_h=gap_res.loc[s].median()))
for lg, g in mk.groupby('lg'):
    e = mk_err.loc[g.index]
    proxy_rows.append(dict(set=f'league={lg}', n_markets=len(e), median_min=e.median(), p05_min=e.quantile(0.05), p25_min=e.quantile(0.25),
                           p75_min=e.quantile(0.75), p95_min=e.quantile(0.95), share_abs_gt_60min=(e.abs() > 60).mean(),
                           share_abs_le_5min=(e.abs() <= 5).mean(), share_close_after_start=(e > 0).mean(),
                           res_minus_start_med_h=gap_res.loc[g.index].median()))
for yr, g in mk.groupby(pd.to_datetime(mk.ind_start, unit='s').dt.year):
    e = mk_err.loc[g.index]
    proxy_rows.append(dict(set=f'year={yr}', n_markets=len(e), median_min=e.median(), share_abs_gt_60min=(e.abs() > 60).mean(),
                           share_abs_le_5min=(e.abs() <= 5).mean(), share_close_after_start=(e > 0).mean()))
proxy = pd.DataFrame(proxy_rows)
proxy.to_csv(f'{R2RES}/T1_proxy_vs_independent.csv', index=False)
print(proxy.round(3).to_string())

# classification changes for orders: before proxy vs before independent start
w = sp[base & sp.ind_ok]
pre_ind = w.ts0 < w.ind_start
pre_px = w.ts0 < w.close_ts
cls = pd.DataFrame([dict(n_orders=len(w), pre_by_independent=pre_ind.mean(), pre_by_proxy=pre_px.mean(),
                         disagree=(pre_ind != pre_px).mean(), proxy_pre_but_independent_post=(pre_px & ~pre_ind).mean(),
                         proxy_post_but_independent_pre=(~pre_px & pre_ind).mean())])
cls.to_csv(f'{R2RES}/T1_classification_change.csv', index=False)
print(cls.round(4).to_string())

# ---- coverage of the sports A sample -------------------------------------------------------------------------------
S0 = sp[base & (sp.res_ts - sp.ts0 >= 6 * 3600)]
cov = []
for lab, g in [('S0 (sample A)', S0)] + [(f'S0, league={lg}', g) for lg, g in S0.groupby(S0.league.where(S0.league.isin(['nba', 'nhl', 'nfl', 'mlb', 'cbb']), 'other'))]:
    cov.append(dict(set=lab, n=len(g), league_in_independent_source=g.lg.isin(['nba', 'nhl', 'nfl']).mean(),
                    slug_parsed=g.lg.notna().mean(), independently_matched=g.ind_ok.mean(),
                    ambiguous=(g.match == 'ambiguous').mean(), unmatched=(g.match == 'unmatched').mean(),
                    old_proxy_available=g.start_ok.mean(),
                    pre_start_among_matched=(g.ts0 < g.ind_start)[g.ind_ok].mean(),
                    pre_start_1h_among_matched=(g.ts0 <= g.ind_start - 3600)[g.ind_ok].mean(),
                    lead_med_h_among_matched=((g.ind_start - g.ts0) / 3600)[g.ind_ok].median()))
cov = pd.DataFrame(cov)
cov.to_csv(f'{R2RES}/T1_coverage.csv', index=False)
print(cov.round(3).to_string())

# ---- samples S0-S3 and outcomes -------------------------------------------------------------------------------------
smp = {'S0 order >= 6h before resolution (current A)': S0,
       'S0 restricted to matched NBA/NHL/NFL': S0[S0.ind_ok],
       'S0 leagues without independent schedule (MLB, CBB, other)': S0[~S0.lg.isin(['nba', 'nhl', 'nfl'])],
       'S1 order before independent scheduled start': sp[base & sp.ind_ok & (sp.ts0 < sp.ind_start)],
       'S1p order before metadata close time (all leagues; close is often 7 days after the start)': sp[base & sp.close_ts.notna() & (sp.ts0 < sp.close_ts)],
       'S2 order >= 1h before independent scheduled start': sp[base & sp.ind_ok & (sp.ts0 <= sp.ind_start - 3600)],
       'S3 S1 and played as scheduled (resolution 1.5-8h after start)': sp[base & sp.ind_ok & (sp.ts0 < sp.ind_start)
                                                                        & ((sp.res_ts - sp.ind_start) / 3600).between(1.5, 8)]}
mem = sp[['w', 'event', 'lg', 'match', 'ind_start', 'mtype']].copy()
for lab, key in [('S0', 'S0 order >= 6h before resolution (current A)'), ('S1', 'S1 order before independent scheduled start'),
                 ('S2', 'S2 order >= 1h before independent scheduled start'),
                 ('S3', 'S3 S1 and played as scheduled (resolution 1.5-8h after start)')]:
    mem[lab] = mem.w.isin(smp[key].w)
mem.to_parquet(f'{WORK}/sp_timing.parquet')
comp = S0.assign(group=np.where(S0.lg.isin(['nba', 'nhl', 'nfl']), 'NBA/NHL/NFL', np.where(S0.mtype == 'nodate', 'undated (series, finals, fights, tournaments)', 'other dated leagues')))
comp.groupby('group').agg(n=('w', 'size'), games=('event', 'nunique'), lead_med_h=('lead', lambda s: s.median() / 3600),
                          lead_p90_h=('lead', lambda s: s.quantile(0.9) / 3600)).to_csv(f'{R2RES}/T1_S0_composition.csv')
out = []
sk = spec('SP_A')
for lab, s in smp.items():
    for y in ['ret7d', 'act22_30', 'act61_90']:
        r = ols2(s, y, **sk, cluster='event')
        if r:
            out.append(dict(sample=lab, n_sample=len(s), n_games=s.event.nunique(), first=pd.to_datetime(s.ts0.min(), unit='s').date(),
                            last=pd.to_datetime(s.ts0.max(), unit='s').date(), win_rate_sample=s.win.mean(),
                            lead_to_start_med_h=((s.ind_start - s.ts0) / 3600).median(), **r))
    r = ols2(s, 'p0', **sk, cluster='event')
    if r:
        out.append(dict(sample=lab, n_sample=len(s), n_games=s.event.nunique(), **{**r, 'y': 'balance_price'}))
out = pd.DataFrame(out)
out.to_csv(f'{R2RES}/T1_sports_timing.csv', index=False)
print(out[['sample', 'y', 'n', 'n_games', 'b', 'se', 'lo', 'hi', 'base', 'first', 'last']].round(4).to_string())
log('done')
