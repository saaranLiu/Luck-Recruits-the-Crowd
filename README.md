# Luck Recruits the Crowd

Data and the script for the current estimates. Crypto sample A is a wallet whose first order is an aggressive, one-sided trade in one up/down market, at a price between 0.30 and 0.70, placed before the price window opens (35,879 wallets). Sports sample S1 applies the same price window and requires that the order precede the scheduled start of an NBA, NHL, or NFL game (123,921 wallets).

The treatment is that the side bought in the first order eventually wins. Outcomes are later fills in any market, counted from the on-chain resolution of that market. Intervals are clustered by market (crypto) or game (sports) and by the UTC day of resolution.

## Contents

| Path | Contents |
|---|---|
| `data/exante.parquet` | The 159,800 analysis wallets |
| `data/sp_timing.parquet` | Scheduled start for each S1 wallet |
| `data/crypto_features.parquet` | Pre-order Binance returns and volatility for the crypto rows used in the rich-control check |
| `data/checks/` | Saved bootstrap, placebo, schedule-flag, and later-trade rows. The script reads them and does not redraw them |
| `estimate.py` | Re-estimates the effects and writes `results/ledger.csv` and `results/curves.csv` |

Field notes are in [`data/README.md`](data/README.md).

## Run

Python 3.10 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run.sh
```

The one-week effects are 3.39 percentage points for crypto and 4.37 points for sports.

An outcome at horizon H is missing when the window would end after 2026-04-28 11:00 UTC. `w` is the address on the public tape. Addresses are not linked to people.
