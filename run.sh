#!/bin/sh
# Re-estimate the reported effects from data/exante.parquet.
set -eu
cd "$(dirname "$0")"
python3 round4/r4_01_export.py
echo "ledger: round4/results/r4_ledger.csv"
echo "curves: round4/results/r4_curves.csv"
