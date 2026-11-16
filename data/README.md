# Analysis files

`exante.parquet` has one row per wallet that enters an estimate: 35,879 crypto wallets and 175,382 sports wallets (211,261 rows). Crypto rows are the price and pre-window sample. Sports rows are those flagged `S0` or `S1` in `sp_timing.parquet`. Wallets that fail those screens are not included. The file is the analysis sample, not a random draw and not the full set of first orders.

`SHA256SUMS` lists every file in this directory.

## Files

| File | What it is |
|---|---|
| `exante.parquet` | Analysis wallets, one row each |
| `sp_timing.parquet` | Match of those sports wallets to an NBA, NHL, or NFL game. `S1` means the order was placed before the scheduled start |
| `quality_r2.parquet` | Later aggressive fills of these wallets in the 90 days after resolution, scored only on markets that had resolved |
| `ud_series.parquet` | Realised up/down outcome of each asset and window length. Used by the time-shift placebo |
| `UD_A_features.parquet`, `UD_C_features.parquet` | Crypto samples with Binance returns and volatility known at the order |
| `klines/binance_15m.parquet` | Binance 15-minute prices used to build those features |
| `schedules/` | NBA (hoopR), NHL (fastRhockey), and NFL (nflverse) games from October 2022 on. Only the columns the match uses: teams, start, status, and game id |
| `markets_sports.parquet` | Condition id and recorded open time for sports markets in `exante.parquet` |

## Fields used by the estimates

| Field | Meaning |
|---|---|
| `w` | Address on the public tape |
| `cid` | Market (`condition_id`) |
| `slug` | Market slug. Sports slugs carry the league, the two teams, and a US Eastern date |
| `ts0` | Block time of the first order, seconds |
| `stake0`, `p0` | Stake and price of the leg bought |
| `win` | 1 if that leg is the one that later pays out |
| `res_ts` | On-chain resolution time. Outcomes are measured from this clock |
| `tstart`, `hz` | Crypto window open and length (`5m`, `15m`, `1h`, `4h`) |
| `n_7d`, `n_30d`, `n_90d` | Fills in any market in (0, H] after resolution |
| `n_w8_14` … `n_w61_90` | Fills in the later windows named by the suffix |
| `days_30`, `weeks_90` | Distinct active days in the first 30 days, and distinct active weeks in the first 13 |
| `t_next` | Seconds from resolution to the next fill. Missing if the wallet does not trade again |
| `n_opp`, `n_maker1`, `n_gap` | Opposite-side fills, passive fills, and fills in another market, all before resolution |

A horizon is missing when `res_ts + horizon` is after 2026-04-28 11:00 UTC.

## What is not here

The fill archive is not in this repository. `12_build_exante.py` and `round2/r2_03_quality_build.py` read it when `POLYMARKET_ROOT` and `MARKETS_PARQUET` are set. Rebuilding `sp_timing.parquet` from the schedules uses the sports rows in `exante.parquet`. This repository does not grant a new license for the Polymarket record or for the league schedules.
