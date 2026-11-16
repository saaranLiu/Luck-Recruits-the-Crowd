"""A1: what the independent schedules can and cannot say about the schedule visible at each S1 order.

Sources (files downloaded 2026-10-10, see A1_schedule_provenance.md):
  NBA  hoopR-nba-data  nba_schedule_master.parquet  (ESPN scoreboard; `date` UTC; `status_type_name`; `game_id`)
  NHL  fastRhockey-nhl-data  nhl_schedule_master.parquet  (NHL API; `game_time` UTC; `game_state`)
  NFL  nflverse nfldata  games.csv  (`gameday` + `gametime`, US Eastern; no status field)
All three are current snapshots: one row per game at its final date and time.  None stores revision timestamps.

Flags are defined from schedule records only (never from effects or win shares):
  F1  the matched schedule row is an NBA game with status Postponed (the market's slug date is the original slot)
  F2  NBA game between a pair that has a Postponed row earlier in the same season (makeup candidates; conservative)
  F3  NBA row without a valid time (time_valid = False)
Diagnostics that use the on-chain resolution time (ex post; reported separately, not used to redefine S1):
  D1  resolution before the scheduled start
  D2  resolution outside 1.5-8 h after the scheduled start (complement of S3 in round 2)
Structural check: the S1 match requires the slug date (fixed when the market is created) to equal the final ET date,
so a game moved to another date after its market was created cannot enter S1 unless the schedule keeps the old slot.
"""
from r3_common import *
import numpy as np, pandas as pd
from r2_samples import load_r2, samples_r2, ols2
from r2_fast import build, ols_fit, cluster_se

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
ALIAS = {'nba': {'nyk': 'ny', 'gsw': 'gs', 'sas': 'sa', 'uta': 'utah', 'nop': 'no', 'was': 'wsh', 'pho': 'phx'},
         'nhl': {'mon': 'mtl', 'tb': 'tbl', 'las': 'vgk', 'utah': 'uta', 'nj': 'njd', 'cal': 'cgy', 'sj': 'sjs'},
         'nfl': {'lar': 'la', 'wsh': 'was', 'las': 'lv'}}

# ---- schedules with status fields ---------------------------------------------------------------------------------
nba = pd.read_parquet(f'{SCHED}/nba_schedule_master.parquet')
nba = pd.DataFrame({'lg': 'nba', 'a': nba.home_abbreviation.str.lower(), 'b': nba.away_abbreviation.str.lower(),
                    'start': pd.to_datetime(nba.date, utc=True), 'status': nba.status_type_name, 'gid': nba.game_id.astype(str),
                    'season': nba.season.astype(str), 'time_valid': nba.time_valid.astype(bool)})
nhl = pd.read_parquet(f'{SCHED}/nhl_schedule_master.parquet')
nhl = pd.DataFrame({'lg': 'nhl', 'a': nhl.home_team_abbr.str.lower(), 'b': nhl.away_team_abbr.str.lower(),
                    'start': pd.to_datetime(nhl.game_time, utc=True), 'status': nhl.game_state, 'gid': nhl.game_id.astype(str),
                    'season': nhl.season.astype(str), 'time_valid': True})
nflr = pd.read_csv(f'{SCHED}/nfl_games.csv')
nflr = nflr[nflr.gametime.notna()]
nfl = pd.DataFrame({'lg': 'nfl', 'a': nflr.home_team.str.lower(), 'b': nflr.away_team.str.lower(),
                    'start': pd.to_datetime(nflr.gameday + ' ' + nflr.gametime).dt.tz_localize('America/New_York').dt.tz_convert('UTC'),
                    'status': np.where(nflr.result.notna(), 'final', 'scheduled'), 'gid': nflr.game_id,
                    'season': nflr.season.astype(str), 'time_valid': True})
sch = pd.concat([nba, nhl, nfl], ignore_index=True)
sch = sch[sch.start >= '2022-10-01'].copy()
sch['date_et'] = sch.start.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
sch['pair'] = [tuple(sorted(x)) for x in zip(sch.a, sch.b)]
sch['start_ts'] = sch.start.astype('int64') // 10**9

# F2: NBA games after a Postponed row of the same pair and season
pp = sch[(sch.lg == 'nba') & (sch.status == 'STATUS_POSTPONED')]
sch['after_pp'] = False
for _, r in pp.iterrows():
    sch.loc[(sch.lg == 'nba') & (sch.season == r.season) & (sch.pair == r.pair) & (sch.start_ts > r.start_ts)
            & (sch.status != 'STATUS_POSTPONED'), 'after_pp'] = True
pp_first = pp.groupby(['season', 'pair']).start_ts.min().rename('pp_ts').reset_index()

prov = pd.DataFrame([
    dict(league='NBA', file='nba_schedule_master.parquet', project='sportsdataverse/hoopR-nba-data', underlying='ESPN scoreboard API',
         downloaded='2026-10-10', rows_since_2022_10=int((sch.lg == 'nba').sum()), time_field='date (UTC, minutes)',
         status_values='; '.join(f'{k}={v}' for k, v in sch[sch.lg == 'nba'].status.value_counts().items()),
         revision_history='no; one row per game_id at its current date/time; postponed slots kept as separate rows',
         rows_time_invalid=int((~sch[sch.lg == 'nba'].time_valid).sum())),
    dict(league='NHL', file='nhl_schedule_master.parquet', project='sportsdataverse/fastRhockey-nhl-data', underlying='NHL web API',
         downloaded='2026-10-10', rows_since_2022_10=int((sch.lg == 'nhl').sum()), time_field='game_time (UTC, minutes)',
         status_values='; '.join(f'{k}={v}' for k, v in sch[sch.lg == 'nhl'].status.value_counts().items()),
         revision_history='no; completed games only, at their final date/time; no postponed rows', rows_time_invalid=0),
    dict(league='NFL', file='games.csv', project='nflverse/nfldata', underlying='nflverse game table',
         downloaded='2026-10-10', rows_since_2022_10=int((sch.lg == 'nfl').sum()), time_field='gameday + gametime (US Eastern, minutes)',
         status_values='no status field (result present = played)',
         revision_history='no; final date/time; cancelled 2022 BUF-CIN game absent', rows_time_invalid=0)])
prov.to_csv(f'{R3RES}/A1_sources.csv', index=False)
print(prov.T.to_string())

# ---- match S1 markets again, keeping status and ids -----------------------------------------------------------------
d = load_r2()
S = samples_r2(d)
s1 = S['SP_S1'].copy()
p = s1.event.str.extract(r'^([a-z]+)-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$')
s1['lg'], s1['t1'], s1['t2'], s1['sdate'] = p[0], p[1], p[2], p[3]
for lg, mp in ALIAS.items():
    for c in ['t1', 't2']:
        s1.loc[s1.lg == lg, c] = s1.loc[s1.lg == lg, c].replace(mp)
s1['pair'] = [tuple(sorted(x)) for x in zip(s1.t1, s1.t2)]
m = s1[['w', 'lg', 'pair', 'sdate']].merge(sch[['lg', 'pair', 'date_et', 'start_ts', 'status', 'gid', 'season', 'time_valid', 'after_pp']],
                                         left_on=['lg', 'pair', 'sdate'], right_on=['lg', 'pair', 'date_et'], how='left')
assert m.w.is_unique and m.gid.notna().all()
assert np.allclose(m.set_index('w').start_ts.loc[s1.w].to_numpy(), s1.ind_start.to_numpy())
s1 = s1.merge(m[['w', 'status', 'gid', 'season', 'time_valid', 'after_pp']], on='w')
s1 = s1.merge(pp_first.assign(lg='nba'), on=['lg', 'season', 'pair'], how='left')

con = connect(threads=16, mem='50GB')
mk = con.sql(f"SELECT condition_id cid, epoch(opens_at) opens_ts FROM '{MARKETS}' WHERE grp = 'SportsML'").df()
s1 = s1.merge(mk, on='cid', how='left')

s1['F1'] = s1.status == 'STATUS_POSTPONED'
s1['F2'] = s1.after_pp.astype(bool)
s1['F3'] = ~s1.time_valid.astype(bool)
gap_h = (s1.res_ts - s1.ind_start) / 3600
s1['D1'] = gap_h < 0
s1['D2'] = ~gap_h.between(1.5, 8)
s1['F2_order_after_pp'] = s1.F2 & (s1.ts0 > s1.pp_ts)
s1['lead_start_h'] = (s1.ind_start - s1.ts0) / 3600
s1['created_before_start_h'] = (s1.ind_start - s1.opens_ts) / 3600

# ---- flag counts ------------------------------------------------------------------------------------------------------
rows = []
N, G = len(s1), s1.event.nunique()
for f, lab in [('F1', 'matched row is a postponed NBA slot'), ('F2', 'NBA game after a postponed game of the same pair and season'),
               ('F2_order_after_pp', '  of which: order placed after the postponed slot'),
               ('F3', 'NBA row without valid time'), ('D1', 'resolution before scheduled start (ex post)'),
               ('D2', 'resolution outside 1.5-8 h after scheduled start (ex post)')]:
    g = s1[s1[f]]
    rows.append(dict(flag=f, label=lab, wallets=len(g), share_wallets=len(g) / N, games=g.event.nunique(), share_games=g.event.nunique() / G,
                     leagues='; '.join(f'{k}={v}' for k, v in g.lg.value_counts().items())))
for lg, g in s1.groupby('lg'):
    rows.append(dict(flag=f'D2|{lg}', label=f'D2 within {lg}', wallets=int(g.D2.sum()), share_wallets=g.D2.mean(),
                     games=g[g.D2].event.nunique(), share_games=g[g.D2].event.nunique() / g.event.nunique()))
any_rec = s1.F1 | s1.F2 | s1.F3
rows.append(dict(flag='F1|F2|F3', label='any schedule-record flag', wallets=int(any_rec.sum()), share_wallets=any_rec.mean(),
                 games=s1[any_rec].event.nunique(), share_games=s1[any_rec].event.nunique() / G))
flags = pd.DataFrame(rows)
flags.to_csv(f'{R3RES}/A1_flags.csv', index=False)
print(flags.round(4).to_string())

# structural facts: market creation versus scheduled start, lead of orders
mkt = s1.drop_duplicates('cid')
facts = dict(s1_wallets=N, s1_games=G, s1_markets=s1.cid.nunique(),
             markets_with_opens_at=int(mkt.opens_ts.notna().sum()),
             market_created_before_start_med_h=mkt.created_before_start_h.median(),
             market_created_before_start_p90_h=mkt.created_before_start_h.quantile(0.9),
             market_created_more_than_7d_before=(mkt.created_before_start_h > 168).mean(),
             order_before_start_med_h=s1.lead_start_h.median(), order_before_start_p90_h=s1.lead_start_h.quantile(0.9),
             orders_more_than_24h_before=(s1.lead_start_h > 24).mean(), orders_more_than_72h_before=(s1.lead_start_h > 72).mean(),
             order_after_market_creation=(s1.ts0 >= s1.opens_ts).mean())
pd.Series(facts).to_csv(f'{R3RES}/A1_structure.csv', header=['value'])
print(pd.Series(facts).round(3).to_string())
detail = s1[s1.F1 | s1.F2 | s1.D1].groupby(['event', 'lg', 'status', 'F1', 'F2', 'D1']).agg(
    wallets=('w', 'size'), sched_start=('ind_start', lambda x: pd.to_datetime(x.iloc[0], unit='s')),
    resolved=('res_ts', lambda x: pd.to_datetime(x.iloc[0], unit='s')), pp_slot=('pp_ts', lambda x: pd.to_datetime(x.iloc[0], unit='s') if x.notna().any() else pd.NaT),
    first_order=('ts0', lambda x: pd.to_datetime(x.min(), unit='s')), last_order=('ts0', lambda x: pd.to_datetime(x.max(), unit='s'))).reset_index()
detail.to_csv(f'{R3RES}/A1_flagged_games.csv', index=False)
print(detail.to_string())

# ---- sensitivity: one-week and 30-day effects excluding flagged games -------------------------------------------------
SP = dict(ctrl=('lstake',), cat=('pbin', 'month', 'hour'))


def est(v, y):
    s = v.dropna(subset=[y, 'lstake']).reset_index(drop=True)
    X, Y, _ = build(s, y, 'win', SP['ctrl'], SP['cat'])
    beta, XtX_inv, e = ols_fit(X, Y)
    se_g, _ = cluster_se(X, e, XtX_inv, s.event.to_numpy())
    se_2, _ = cluster_se(X, e, XtX_inv, [s.event.to_numpy(), s.rday.to_numpy()])
    return dict(y=y, n=len(s), games=s.event.nunique(), b=100 * beta[0], se_game=100 * se_g, se_2way_rday=100 * se_2,
                lo_game=100 * (beta[0] - 1.96 * se_g), hi_game=100 * (beta[0] + 1.96 * se_g),
                lo_2way=100 * (beta[0] - 1.96 * se_2), hi_2way=100 * (beta[0] + 1.96 * se_2))


sens = []
for lab, keep, src in [('current S1', np.ones(N, bool), 'independent schedule, slug date = final ET date'),
                       ('excl. F1 (postponed NBA slots)', ~s1.F1, 'schedule record: status Postponed'),
                       ('excl. F1-F3 (all schedule-record flags)', ~any_rec, 'schedule record: postponed slots and later games of the same pair'),
                       ('excl. F1-F3 and D2 (resolution 1.5-8 h after start)', ~any_rec & ~s1.D2, 'adds an ex-post resolution-time rule (diagnostic)')]:
    v = s1[keep.to_numpy() if hasattr(keep, 'to_numpy') else keep]
    for y in ['ret7d', 'ret30d']:
        sens.append(dict(sample=lab, wallets_in_sample=len(v), excluded_share=1 - len(v) / N, timing_source=src, **est(v, y)))
sens = pd.DataFrame(sens)
sens.to_csv(f'{R3RES}/A1_sensitivity.csv', index=False)
print(sens.round(3).to_string())
log('done')
