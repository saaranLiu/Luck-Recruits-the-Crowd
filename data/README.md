# Analysis files

One row of `exante.parquet` is one wallet. The row is kept when the wallet's first fill is an aggressive order in a single resolved market, on one side, in a crypto up/down market or a sports moneyline market. Later sells, hedges, and trades in other markets are stored on the row. They are not used to decide whether the wallet is eligible.

`SHA256SUMS` lists every file in this directory. `sp_timing.parquet` and `quality_r2.parquet` are also copied under `round2/work/` and `round3/pipeline/work/` so the scripts can keep their original relative paths. Those copies are the same bytes.

## Files

| File | What it is |
|---|---|
| `exante.parquet` | 627,474 wallets. Crypto and sports candidates, before the price and timing restrictions |
| `sp_timing.parquet` | Sports wallets matched to an NBA, NHL, or NFL game. `S1` is an order placed before the scheduled start |
| `quality_r2.parquet` | Later aggressive fills in the 90 days after resolution, scored only on markets that had resolved by the end of the tape |
| `ud_series.parquet` | Realised up/down outcome of each asset and window length. Used by the time-shift placebo |
| `UD_A_features.parquet`, `UD_C_features.parquet` | Crypto samples with Binance returns and volatility known at the order |
| `klines/binance_15m.parquet` | Binance 15-minute prices used to build those features |
| `schedules/` | Final NBA (hoopR), NHL (fastRhockey), and NFL (nflverse) schedules. A postponed game appears at its final time |
| `markets_sports.parquet` | Sports market id and the recorded open time. The schedule audit reads this file |

## Fields used by the estimates

| Field | Meaning |
|---|---|
| `w` | Wallet address, as on chain |
| `cid` | Market (`condition_id`) |
| `slug` | Market slug. Sports slugs carry the league, the two teams, and a US Eastern date |
| `ts0` | Block time of the first order, seconds |
| `stake0`, `p0` | Stake and price of the leg bought, in USDC and probability units |
| `win` | 1 if that leg is the one that later pays out |
| `res_ts` | On-chain resolution time. Outcomes are measured from this clock |
| `tstart`, `hz` | Crypto window open and length (`5m`, `15m`, `1h`, `4h`) |
| `n_7d`, `n_30d`, `n_90d` | Fills in any market in (0, H] after resolution |
| `n_w8_14` … `n_w61_90` | Fills in the later windows named by the suffix |
| `days_30`, `weeks_90` | Distinct active days in the first 30 days, and distinct active weeks in the first 13 |
| `t_next` | Seconds from resolution to the next fill. Missing if the wallet does not trade again in the tape |
| `n_opp`, `n_maker1`, `n_gap` | Opposite-side fills, passive fills, and fills in another market, all before resolution |

The crypto sample in the paper is `grp = UpDown`, `p0` in [0.3, 0.7], and `ts0` before `tstart` (35,879 wallets). The sports sample is `S1` with the same price window (123,921 wallets). A horizon is missing when `res_ts + horizon` is after 2026-04-28 11:00 UTC.

## What is not here

The fill archive is not in this repository. `12_build_exante.py` and `round2/r2_03_quality_build.py` read it when `POLYMARKET_ROOT` and `MARKETS_PARQUET` are set. Prices, stakes, and wallet addresses come from that public tape. This repository does not grant a new license for the Polymarket record or for the league schedules.
