# Analysis files

`exante.parquet` has one row for each wallet in the current samples: 35,879 crypto and 123,921 sports. `SHA256SUMS` lists the files in this directory.

| File | What it is |
|---|---|
| `exante.parquet` | Analysis wallets |
| `sp_timing.parquet` | `ind_start`, the scheduled start in seconds, and `S1` for the sports wallets |
| `crypto_features.parquet` | Crypto rows with Binance returns and volatility known at the order |
| `checks/` | Saved rows for the bootstrap intervals, the time-shift placebo, the sports simulated null, the NBA schedule flag, and the later-trade comparison |

| Field | Meaning |
|---|---|
| `w` | Address on the public tape |
| `cid` | Market |
| `slug` | Market slug. Sports slugs carry the league, the two teams, and a US Eastern date |
| `ts0` | Block time of the first order, seconds |
| `stake0`, `p0` | Stake and price of the leg bought |
| `win` | 1 if that leg later pays out |
| `res_ts` | On-chain resolution time |
| `tstart`, `hz` | Crypto window open and length (`5m`, `15m`, `1h`, `4h`) |
| `n_7d`, `n_30d`, `n_90d` | Fills in any market in (0, H] after resolution |
| `n_w8_14` … `n_w61_90` | Fills in the later windows named by the suffix |
| `days_30`, `weeks_90` | Distinct active days in 30 days, and distinct active weeks in 13 weeks |
| `t_next` | Seconds from resolution to the next fill |

The fill archive and the league schedule files are not in this repository. This repository does not grant a new license for the Polymarket record.
