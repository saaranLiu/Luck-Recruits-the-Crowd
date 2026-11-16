# Luck Recruits the Crowd

Wallet-level data and the scripts that re-estimate the reported effects. A wallet is in the crypto sample when its first order is an aggressive, one-sided trade in a single up/down market, at a price between 0.30 and 0.70, placed before the price window opens. A wallet is in the sports sample when the same price condition holds and the order is placed before the scheduled start of an NBA, NHL, or NFL game. Both samples are fixed at that first order.

The treatment is that the side bought in the first order is the side that eventually wins. Outcomes are later fills in any market, counted from the on-chain resolution of the first market. Intervals are clustered two ways, by market (crypto) or game (sports) and by the UTC day of resolution.

This repository has no manuscript and no typesetting code. It does not include the raw fill archive.

## Contents

| Path | Contents |
|---|---|
| [`data/`](data/) | Analysis wallets, the sports schedule match, later-trade scores, Binance prices, and the schedule columns used for the match. Notes are in [`data/README.md`](data/README.md) |
| [`round4/r4_01_export.py`](round4/r4_01_export.py) | Re-estimates the reported effects and the return curves |
| [`round3/`](round3/) | Schedule flags, timing checks, persistence, placebo and simulated-null ranks, later-trade comparisons |
| [`round2/`](round2/) | How the schedule match and the later-trade file are built |
| `14_random.py` | Out-of-sample prediction of the first outcome from information known at the order |
| `12_build_exante.py` | Rebuilds the wallet file from the fill tape. The tape is not included |

## Re-estimate

Python 3.10 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run.sh
```

`run.sh` reads `data/exante.parquet` and the files it joins, and writes `round4/results/r4_ledger.csv` and `r4_curves.csv`. The one-week effects are 3.39 percentage points for crypto (35,879 wallets) and 4.37 points for sports (123,921 wallets).

Bootstrap intervals, the time-shift placebo, the sports simulated null, the schedule-flag row, and the later-trade comparison are copied from the saved round-3 files. Those files can be rebuilt with the commands below. The redraws use 999 or 2,000 replicates.

```sh
python3 14_random.py
python3 round2/r2_01_timing.py
cd round3
python3 r3_01_schedule_audit.py
python3 r3_02_time_controls.py
cd pipeline
python3 r2_05_quality.py
python3 r2_06_persistence.py
python3 r2_07_nulls.py
python3 r2_08_estimand.py
python3 r2_09_main.py
python3 r3_03_ud_extras.py
python3 r3_04_cluster100.py
```

`r2_06_persistence.py` and `r2_07_nulls.py` write the bootstrap and null rows that `round4/r4_01_export.py` copies into the ledger. `r2_09_main.py` writes the sports prediction rows. `14_random.py` writes `results/rand_prediction.csv`.

## Rebuild the wallet file from the fill tape

The tape is the public Polymarket record: `database/daily_aligned/`, `database/OrderFilled/`, and `database/CTF/`, plus the full market table. None of that is in this repository. `data/exante.parquet` already contains every wallet the estimates use.

```sh
export POLYMARKET_ROOT=/path/to/archive
export MARKETS_PARQUET=/path/to/markets.parquet
export ANALYSIS_THREADS=32
export ANALYSIS_MEMORY=600GB
python3 12_build_exante.py
python3 round2/r2_03_quality_build.py
```

`data/markets_sports.parquet` is only the open time of sports markets in the analysis file. It is not a substitute for `MARKETS_PARQUET`.

## Measurement

- A first-bet win is the experience that the bought side won, including whatever the wallet does before resolution. It is not the receipt of a payout.
- An outcome at horizon H is defined only when the window ends by 2026-04-28 11:00 UTC. The common 90-day comparison uses the 25,222 crypto and 92,706 sports wallets for which that is true.
- Sports start times are the final published schedules. A game moved to another date does not match. Same-day changes, and NHL or NFL rescheduling, are not identified. The NBA postponed flag is the check in `round3/r3_01_schedule_audit.py`.
- `w` is the address on the public tape. Addresses are not linked to people.

Saved checks are in `round3/results/`, `round3/pipeline/results/`, and `results/`. Third-party files and their limits are in [`data/README.md`](data/README.md).
