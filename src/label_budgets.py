"""The label budgets of the label-efficiency curve, shared by run_label_curve.py (AnyJev L2, torch)
and run_label_curve_tabular.py (LightGBM / logistic regression). They run as separate processes:
torch and LightGBM each bring their own OpenMP runtime, and loading both into one process
segfaults on macOS."""
import pandas as pd

from llm_common import load_split
from prepare_data import stratified_sample

REPEATS = {25: 5, 50: 5, 100: 5, 300: 3, 1000: 1}
SMOKE_REPEATS = {10: 2, 25: 2}


def budgets(cfg):
    return SMOKE_REPEATS if cfg["smoke_mode"] else REPEATS


def pools(cfg):
    """One ordered pool per repeat; budget n takes the first n rows (roughly stratified).
    Repeat 0 starts with the exact 300 bookings of the main L1 / L2 / low-data runs."""
    reps, seed = budgets(cfg), cfg["seed"]
    calib = load_split(cfg, "calib")
    base = load_split(cfg, "calib_sample")
    rest = calib[~calib.booking_id.isin(base.booking_id)].sample(frac=1.0, random_state=seed)
    out = [pd.concat([base, rest]).head(max(reps))]
    for r in range(1, max(reps.values())):
        out.append(stratified_sample(calib, min(max(reps), 300), seed + r))
    return out


def draws(cfg):
    """(n_labels, repeat, labelled rows) for every fit, in a fixed order."""
    orders = pools(cfg)
    return [(n, r, orders[r].head(n)) for n, k in budgets(cfg).items() for r in range(k)]
