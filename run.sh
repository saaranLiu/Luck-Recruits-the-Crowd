#!/bin/sh
# Re-estimate the current crypto A and sports S1 effects.
set -eu
cd "$(dirname "$0")"
python3 estimate.py
echo "ledger: results/ledger.csv"
echo "curves: results/curves.csv"
