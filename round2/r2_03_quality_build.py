"""T2/T4: later-bet metrics with the round-2 definitions, for sample wallets (up/down A, sports S0 and S1).

Later fills: aggressive (taker) fills in the 90 days after the first market's resolution, (res_ts, res_ts + 90 d].
Per fill: n = usdc_amount / price, c = n * q, e = D (Y - p_event), fee = fee_usdc (taker fee, observed on chain).
Per wallet, resolved markets only:
  edge_share  = sum n e / sum n            gross P&L per contract traded (invariant to Y for closed or hedged positions)
  edge_cost   = sum c e / sum c            round-1 cost-weighted contract edge (kept for comparison)
  ret_turn    = (sum n e - sum fee) / sum c   net P&L per dollar traded (turnover-based, not return on capital)
  win_leg     = mean 1[e > 0]              share of fills on the leg that eventually won
Coverage: passive (maker) later fills and fills in markets unresolved at the end of the data are counted, not scored.
"""
from r2_common import *
import pandas as pd
from r2_samples import load_r2, samples_r2

if not RAW_TAPE:
    raise SystemExit(
        "Rebuilding data/quality_r2.parquet needs the Polymarket fill tape. "
        "Set POLYMARKET_ROOT and MARKETS_PARQUET. See README.md."
    )

d = load_r2()
S = samples_r2(d)
ws = pd.concat([S['UD_A'], S['SP_S0'], S['SP_S1']]).drop_duplicates('w')[['w', 'res_ts']]
con = connect(threads=128, mem='500GB')
con.register('ws_df', ws)
con.execute('CREATE TABLE ws AS SELECT * FROM ws_df')
log('wallets', len(ws))
q = con.sql(f"""
WITH f AS (
  SELECT ws.w, d.condition_id cid, d.D, d.p_event p, d.usdc_amount / d.price n,
         d.usdc_amount / d.price * CASE WHEN d.D = 1 THEN d.p_event ELSE 1 - d.p_event END c,
         coalesce(d.fee_usdc, 0) fee, 1 is_t
  FROM read_parquet('{DA}') d JOIN ws ON d.taker = ws.w
  WHERE d.block_timestamp > ws.res_ts AND d.block_timestamp <= ws.res_ts + 90 * {DAY} AND d.price > 0 AND d.price < 1
  UNION ALL
  SELECT ws.w, d.condition_id, -d.D, d.p_event, d.usdc_amount / d.price,
         d.usdc_amount / d.price * CASE WHEN d.D = -1 THEN d.p_event ELSE 1 - d.p_event END, 0, 0
  FROM read_parquet('{DA}') d JOIN ws ON d.maker = ws.w
  WHERE d.block_timestamp > ws.res_ts AND d.block_timestamp <= ws.res_ts + 90 * {DAY} AND d.price > 0 AND d.price < 1
), g AS (SELECT f.*, m.Y, f.D * (m.Y - f.p) e FROM f LEFT JOIN '{MARKETS}' m ON m.condition_id = f.cid)
SELECT w,
  count(*) FILTER (WHERE is_t = 1 AND Y IS NOT NULL) r_n,
  count(DISTINCT cid) FILTER (WHERE is_t = 1 AND Y IS NOT NULL) r_mk,
  sum(n) FILTER (WHERE is_t = 1 AND Y IS NOT NULL) r_sh,
  sum(c) FILTER (WHERE is_t = 1 AND Y IS NOT NULL) r_c,
  sum(n * e) FILTER (WHERE is_t = 1 AND Y IS NOT NULL) r_ne,
  sum(c * e) FILTER (WHERE is_t = 1 AND Y IS NOT NULL) r_ce,
  sum(fee) FILTER (WHERE is_t = 1 AND Y IS NOT NULL) r_fee,
  avg((e > 0)::INT) FILTER (WHERE is_t = 1 AND Y IS NOT NULL) r_win_leg,
  sum(c) FILTER (WHERE is_t = 1 AND Y IS NULL) r_c_unres,
  sum(c) FILTER (WHERE is_t = 0) r_c_passive,
  count(*) FILTER (WHERE is_t = 0) r_n_passive
FROM g GROUP BY w
""").df()
q['edge_share'] = q.r_ne / q.r_sh
q['edge_cost'] = q.r_ce / q.r_c
q['ret_turn'] = (q.r_ne - q.r_fee) / q.r_c
q.to_parquet(f'{R2WORK}/quality_r2.parquet')
log('saved', len(q), 'with scored fills', (q.r_n > 0).sum())
