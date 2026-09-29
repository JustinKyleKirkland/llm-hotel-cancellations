#!/usr/bin/env bash
# Run the whole pipeline in order.   ./run_all.sh          full run (config.yaml: n_test bookings)
#                                     ./run_all.sh --smoke  50 bookings, a few minutes
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
FLAG=${1:-}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
step() { echo; echo "=== $1 ==="; shift; "$PY" "$@" $FLAG; }
step "1/15 data"             src/prepare_data.py
step "2/15 example prompts"  src/booking_text.py
step "3/15 classical (full)" src/run_classical.py
step "4/15 low-data (300)"   src/run_lowdata.py
step "5/15 naive LLM"        src/run_naive.py
step "6/15 AnyJev L0"        src/run_l0.py
step "7/15 AnyJev L1"        src/run_l1.py
step "8/15 AnyJev L2 (extra)" src/run_l2.py
step "9/15 L0 + one-sentence hint" src/run_l0_hint.py
step "10/15 L0 prior strength" src/run_prior_strength.py
step "11/15 order flip"       src/run_order_flip.py
step "12/15 label curve: L2" src/run_label_curve.py
step "13/15 label curve: tabular" src/run_label_curve_tabular.py
step "14/15 metrics"          src/evaluate.py
step "15/15 figures"         src/make_figures.py
echo; echo "done. outputs in outputs/${FLAG:+smoke/}, figures in figures/${FLAG:+smoke/}"
