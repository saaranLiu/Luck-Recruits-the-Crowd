#!/bin/sh
# Re-estimate the short-paper ledger, then write the figures and the two tables.
set -eu
cd "$(dirname "$0")"
python3 round4/r4_01_export.py
python3 round4/r4_02_figures.py
python3 round4/r4_03_table.py
echo "ledger:  round4/results/r4_ledger.csv"
echo "curves:  round4/results/r4_curves.csv"
echo "figures: round4/figures/"
echo "tables:  results/tables/"
