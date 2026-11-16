# Luck Recruits the Crowd

Analysis data and scripts for the estimates on first-time Polymarket wallets. A wallet enters the crypto sample when its first order is an aggressive, one-sided trade in a single up/down market, at a price between 0.30 and 0.70, placed before the price window opens. A wallet enters the sports sample when the same order conditions hold and the order is placed before the scheduled start of an NBA, NHL, or NFL game. The samples are fixed at that first order.

The treatment is that the side bought in the first order is the side that eventually wins. Outcomes are later fills in any market, counted from the on-chain resolution of the first market. Standard errors in the short-paper tables are clustered two ways, by market (crypto) or game (sports) and by the UTC day of resolution.

This repository does not include a manuscript. It does not include the raw fill archive.

## Contents

| Path | Contents |
|---|---|
| [`data/`](data/) | Wallet file, sports schedule match, later-trade scores, Binance prices, and the NBA, NHL, and NFL schedules. Field notes are in [`data/README.md`](data/README.md) |
| [`round4/`](round4/) | Short-paper estimates, figures, and the two tables |
| [`round3/`](round3/) | Schedule flags, timing-control checks, persistence, placebo and simulated-null ranks, and later-trade comparisons |
| [`round2/`](round2/) | Schedule match (`r2_01_timing.py`) and the construction of the later-trade file |
| `14_random.py` | Out-of-sample prediction of the first outcome from information known at the order |
| `12_build_exante.py` | Rebuilds `data/exante.parquet` from the fill tape. The tape is not included |

Figure labels use Linux Biolinum (SIL Open Font License); the files are in `round4/fonts/`.

## Reproduce the short-paper tables and figures

Python 3.10 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run.sh
```

`run.sh` re-estimates the ledger from `data/exante.parquet` and the files it joins, then writes:

- `round4/results/r4_ledger.csv` and `r4_curves.csv`
- `round4/figures/fig1_design.pdf`, `fig2_random.pdf`, `fig3_dynamic.pdf`, `fig4_bounds.pdf`
- `results/tables/tab1_main.tex` and `tab2_dynamic.tex`

The one-week estimates this produces are 3.39 percentage points for crypto (35,879 wallets) and 4.37 points for sports (123,921 wallets). Bootstrap intervals, the time-shift placebo, the sports simulated null, the schedule-flag row, and the later-trade comparison are read from the saved round-3 files rather than redrawn. Those files can be rebuilt with the commands below. The redraws use 999 or 2,000 replicates and take substantially longer than `run.sh`.

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

`r2_06_persistence.py` and `r2_07_nulls.py` write the bootstrap and null rows that `round4/r4_01_export.py` copies into the ledger. `r2_09_main.py` writes the sports prediction panel. `14_random.py` writes `results/rand_prediction.csv`.

## Rebuild the wallet file from the fill tape

The tape is the Polymarket-v1 record: `database/daily_aligned/`, `database/OrderFilled/`, and `database/CTF/`, plus the market table. It is not distributed here.

```sh
export POLYMARKET_ROOT=/path/to/archive
export MARKETS_PARQUET=/path/to/markets.parquet
export ANALYSIS_THREADS=32
export ANALYSIS_MEMORY=600GB
python3 12_build_exante.py
python3 round2/r2_03_quality_build.py
```

`data/markets_sports.parquet` is only the sports open times used by the schedule audit. It is not a substitute for `MARKETS_PARQUET`.

## Measurement

- A settlement of the first bet is the experience that the bought side won, including whatever the wallet does before resolution. It is not the receipt of a payout.
- An outcome at horizon H is defined only when the window ends by 2026-04-28 11:00 UTC. The common 90-day sample is the 25,222 crypto and 92,706 sports wallets for which that is true.
- Sports start times are the final published schedules. A game moved to another date does not match. Same-day changes, and NHL or NFL rescheduling, are not identified. The NBA postponed flag is the check in `round3/r3_01_schedule_audit.py`.
- Wallet addresses are the addresses on the public tape. They are not linked to people.

Saved numerical checks are in `round3/results/`, `round3/pipeline/results/`, and `results/`. Sources and the limits on third-party files are in [`data/README.md`](data/README.md).
