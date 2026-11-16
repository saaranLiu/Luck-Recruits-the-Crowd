"""Ex-ante sample (A) and its nested restrictions (B: held to resolution, C: no other trading before resolution).

Every inclusion rule uses only what is fixed when the first order executes: the wallet's first fill is aggressive and
in a single market, all fills of that first transaction take the same event-normalised side D, and its price and timing
are known.  A wallet holds no tokens before its first fill, so the first transaction is a purchase of leg D (in a mint
match the raw tape labels it a SELL of the opposite token and usdc_amount is the counterparty's notional).  Hence
shares = usdc_amount / price (the row token's price) and the cost of a fill = shares * q, q = p_event if D = 1 else
1 - p_event, the price of the leg actually acquired.
Treatment = the leg bought by the first order wins.  Later behaviour in the first market (adding, selling, hedging)
and trading elsewhere before resolution are recorded as intermediate outcomes, not used to select the sample.
Output: data/exante.parquet (one row per wallet).
"""
from common import *

if not RAW_TAPE:
    raise SystemExit(
        "Rebuilding data/exante.parquet needs the Polymarket fill tape. "
        "Set POLYMARKET_ROOT and MARKETS_PARQUET. See README.md."
    )

con = connect()
log('start')
con.execute(f"""
CREATE TABLE fills AS
SELECT taker AS w, block_timestamp ts, condition_id cid, 1::TINYINT is_t, usdc_amount / price sh,
       usdc_amount / price * CASE WHEN D = 1 THEN p_event ELSE 1 - p_event END usd, D::TINYINT D, p_event p,
       coalesce(fee_usdc, 0) fee
FROM read_parquet('{DA}') WHERE price > 0 AND price < 1 AND p_event > 0 AND p_event < 1
UNION ALL
SELECT maker AS w, block_timestamp ts, condition_id cid, 0::TINYINT is_t, usdc_amount / price sh,
       usdc_amount / price * CASE WHEN D = -1 THEN p_event ELSE 1 - p_event END usd, (-D)::TINYINT D, p_event p, 0.0 fee
FROM read_parquet('{DA}') WHERE price > 0 AND price < 1 AND p_event > 0 AND p_event < 1
""")
log('fills', con.sql('SELECT count(*) FROM fills').fetchone())

# ---- first transaction of every wallet ----------------------------------------------------------------------
con.execute("""
CREATE TABLE first AS
WITH f0 AS (SELECT w, min(ts) t0 FROM fills GROUP BY w)
SELECT f.w, f0.t0, count(DISTINCT f.cid) n_cid0, min(f.is_t) all_taker0,
       min(f.D) dmin0, max(f.D) dmax0, any_value(f.cid) cid, any_value(f.D) D0, count(*) nf0,
       sum(f.usd) stake0, sum(f.sh) shares0, sum(f.fee) fee0
FROM fills f JOIN f0 ON f.w = f0.w AND f.ts = f0.t0
GROUP BY f.w, f0.t0
""")
log('first', con.sql("SELECT count(*), sum((n_cid0=1 AND all_taker0=1 AND dmin0=dmax0)::INT) FROM first").fetchone())

con.execute(f"""
CREATE TABLE cand AS
SELECT fi.w, fi.cid, fi.t0 ts0, fi.D0, fi.nf0, fi.stake0, fi.shares0, fi.fee0, fi.stake0 / fi.shares0 p0,
       m.grp, m.slug, m.ref_label, m.Y, m.res_ts::DOUBLE res_ts, epoch(m.close_at) close_ts,
       CASE WHEN (fi.D0 = 1 AND m.Y = 1) OR (fi.D0 = -1 AND m.Y = 0) THEN 1 ELSE 0 END win
FROM first fi JOIN '{MARKETS}' m ON m.condition_id = fi.cid
WHERE fi.n_cid0 = 1 AND fi.all_taker0 = 1 AND fi.dmin0 = fi.dmax0
  AND m.grp IN ('UpDown', 'SportsML') AND m.Y IS NOT NULL AND m.res_ts IS NOT NULL AND fi.t0 < m.res_ts::DOUBLE
""")
log('candidates', con.sql('SELECT grp, count(*) FROM cand GROUP BY 1').fetchall())

con.execute('CREATE TABLE act AS SELECT f.* FROM fills f SEMI JOIN cand c ON f.w = c.w')
con.execute('DROP TABLE fills')
log('act', con.sql('SELECT count(*) FROM act').fetchone())

# ---- behaviour between the first order and resolution (intermediate outcomes) --------------------------------
con.execute("""
CREATE TABLE mid AS
SELECT c.w,
  count(*) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts) n_first_mkt,
  count(*) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts AND a.ts > c.ts0 AND a.D = c.D0) n_add,
  count(*) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts AND a.D <> c.D0) n_opp,
  count(*) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts AND a.is_t = 0) n_maker1,
  max(a.ts) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts) ts1,
  sum(CASE WHEN a.D = c.D0 THEN 1 ELSE -1 END * a.sh) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts) net_sh,
  sum(a.sh) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts AND a.D = c.D0) gross_sh,
  sum(a.usd) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts AND a.D = c.D0) stake_all,
  min(a.ts) FILTER (WHERE a.cid = c.cid AND a.ts <= c.res_ts AND a.D <> c.D0) t_first_opp,
  count(*) FILTER (WHERE a.cid <> c.cid AND a.ts > c.ts0 AND a.ts <= c.res_ts) n_gap,
  min(a.ts) FILTER (WHERE a.cid <> c.cid AND a.ts > c.ts0 AND a.ts <= c.res_ts) t_first_gap
FROM cand c LEFT JOIN act a ON a.w = c.w
GROUP BY c.w
""")
log('mid done')

# ---- activity after resolution ----------------------------------------------------------------------------------
con.execute(f"""
CREATE TABLE post AS
WITH x AS (SELECT c.w, a.ts - c.res_ts dt, a.is_t, a.usd, a.cid FROM cand c JOIN act a ON a.w = c.w WHERE a.ts > c.res_ts)
SELECT w, min(dt) t_next,
  count(*) FILTER (WHERE dt <= {DAY}) n_1d,
  count(*) FILTER (WHERE dt <= 7*{DAY}) n_7d,
  count(*) FILTER (WHERE dt <= 14*{DAY}) n_14d,
  count(*) FILTER (WHERE dt <= 30*{DAY}) n_30d,
  count(*) FILTER (WHERE dt <= 60*{DAY}) n_60d,
  count(*) FILTER (WHERE dt <= 90*{DAY}) n_90d,
  count(*) FILTER (WHERE dt > 7*{DAY} AND dt <= 14*{DAY}) n_w8_14,
  count(*) FILTER (WHERE dt > 14*{DAY} AND dt <= 21*{DAY}) n_w15_21,
  count(*) FILTER (WHERE dt > 21*{DAY} AND dt <= 30*{DAY}) n_w22_30,
  count(*) FILTER (WHERE dt > 30*{DAY} AND dt <= 60*{DAY}) n_w31_60,
  count(*) FILTER (WHERE dt > 60*{DAY} AND dt <= 90*{DAY}) n_w61_90,
  count(DISTINCT floor(dt / {DAY})) FILTER (WHERE dt <= 30*{DAY}) days_30,
  count(DISTINCT floor(dt / (7*{DAY}))) FILTER (WHERE dt <= 90*{DAY}) weeks_90,
  coalesce(sum(usd) FILTER (WHERE dt <= 30*{DAY} AND is_t = 1), 0) usd_30d,
  count(DISTINCT cid) FILTER (WHERE dt <= 30*{DAY}) mk_30d,
  count(*) n_all
FROM x GROUP BY w
""")
log('post done')

# ---- quality of later aggressive trades (90 days, resolved markets) ---------------------------------------------
# per fill j: contract edge e_j = D_j (Y - p_j) = payoff minus price of the leg acquired (a sale of a held leg at p is
# scored as forgoing its payoff); dollar edge = shares_j * e_j - fee_j.
con.execute(f"""
CREATE TABLE qual AS
SELECT c.w,
  count(*) FILTER (WHERE m.Y IS NOT NULL) q_n,
  sum(a.usd) FILTER (WHERE m.Y IS NOT NULL) q_usd,
  sum(a.usd) FILTER (WHERE m.Y IS NULL) q_usd_unres,
  avg(a.D * (m.Y - a.p)) q_edge,
  sum(a.usd * a.D * (m.Y - a.p)) / sum(a.usd) FILTER (WHERE m.Y IS NOT NULL) q_edge_usd,
  sum(a.sh * a.D * (m.Y - a.p) - a.fee) FILTER (WHERE m.Y IS NOT NULL) q_pnl,
  sum(a.fee) FILTER (WHERE m.Y IS NOT NULL) q_fee,
  avg(CASE WHEN (a.D = 1 AND m.Y = 1) OR (a.D = -1 AND m.Y = 0) THEN 1 ELSE 0 END) FILTER (WHERE m.Y IS NOT NULL) q_hit,
  count(DISTINCT a.cid) FILTER (WHERE m.Y IS NOT NULL) q_mk
FROM cand c JOIN act a ON a.w = c.w LEFT JOIN '{MARKETS}' m ON m.condition_id = a.cid
WHERE a.is_t = 1 AND a.ts > c.res_ts AND a.ts <= c.res_ts + 90*{DAY}
GROUP BY c.w
""")
log('qual done')

# ---- next aggressive trade after resolution ----------------------------------------------------------------------
con.execute(f"""
CREATE TABLE nxt AS
WITH t AS (
  SELECT a.w, a.cid, a.ts, sum(a.usd) usd, sum(a.usd) / sum(a.sh) pi
  FROM act a JOIN cand c ON a.w = c.w
  WHERE a.is_t = 1 AND a.ts > c.res_ts
  GROUP BY a.w, a.cid, a.ts
), r AS (SELECT *, row_number() OVER (PARTITION BY w ORDER BY ts, cid) k FROM t)
SELECT w, ts ts_next_bet, usd stake2, pi pi2 FROM r WHERE k = 1
""")

# ---- up/down series: asset, horizon, window start, and neighbouring windows for placebos -----------------------
con.execute(f"""
CREATE TABLE ud AS
SELECT condition_id cid,
       regexp_extract(slug, '^([a-z]+)-updown-(5m|15m|1h|4h)-([0-9]+)$', 1) asset,
       regexp_extract(slug, '^([a-z]+)-updown-(5m|15m|1h|4h)-([0-9]+)$', 2) hz,
       regexp_extract(slug, '^([a-z]+)-updown-(5m|15m|1h|4h)-([0-9]+)$', 3)::BIGINT tstart,
       CASE WHEN lower(ref_label) = 'up' THEN Y ELSE 1 - Y END up
FROM '{MARKETS}' WHERE grp = 'UpDown' AND regexp_matches(slug, '^[a-z]+-updown-(5m|15m|1h|4h)-[0-9]+$') AND Y IS NOT NULL
""")
con.execute("""
CREATE TABLE udp AS
SELECT u.cid, u.asset, u.hz, u.tstart, u.up, e.up up_m1h, l.up up_p1h
FROM ud u LEFT JOIN ud e ON e.asset = u.asset AND e.hz = u.hz AND e.tstart = u.tstart - 3600
          LEFT JOIN ud l ON l.asset = u.asset AND l.hz = u.hz AND l.tstart = u.tstart + 3600
""")
con.execute(f"COPY ud TO '{WORK}/ud_series.parquet' (FORMAT parquet)")

con.execute(f"""
COPY (
  SELECT c.*, mi.* EXCLUDE (w), p.* EXCLUDE (w), q.* EXCLUDE (w), n.* EXCLUDE (w),
         u.asset, u.hz, u.tstart, u.up, u.up_m1h, u.up_p1h,
         CASE WHEN c.grp = 'UpDown' THEN ((c.D0 = 1) = (lower(c.ref_label) = 'up'))::INT END side_up
  FROM cand c JOIN mid mi USING (w) LEFT JOIN post p USING (w) LEFT JOIN qual q USING (w) LEFT JOIN nxt n USING (w)
       LEFT JOIN udp u ON u.cid = c.cid
) TO '{WORK}/exante.parquet' (FORMAT parquet)
""")
log('saved', con.sql(f"SELECT grp, count(*), avg(win) FROM '{WORK}/exante.parquet' GROUP BY 1").fetchall())
