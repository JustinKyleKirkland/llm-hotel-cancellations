#!/usr/bin/env bash
# Run the whole pipeline in order.   ./run_all.sh          full run (config.yaml: n_test bookings)
#                                     ./run_all.sh --smoke  50 bookings, a few minutes
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
FLAG=${1:-}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
step() { echo; echo "=== $1 ==="; shift; "$PY" "$@" $FLAG; }
step "1/11 data"             src/prepare_data.py
step "2/11 example prompts"  src/booking_text.py
step "3/11 classical (full)" src/run_classical.py
step "4/11 low-data (300)"   src/run_lowdata.py
step "5/11 naive LLM"        src/run_naive.py
step "6/11 AnyJev L0"        src/run_l0.py
step "7/11 AnyJev L1"        src/run_l1.py
step "8/11 AnyJev L2 (extra)" src/run_l2.py
step "9/11 order flip"       src/run_order_flip.py
step "10/11 metrics"          src/evaluate.py
step "11/11 figures"         src/make_figures.py
echo; echo "done. outputs in outputs/${FLAG:+smoke/}, figures in figures/${FLAG:+smoke/}"
