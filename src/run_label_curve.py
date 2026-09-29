"""Label-efficiency curve: how accurate does each method get with 25, 50, 100, 300 or 1,000 labels?

For each budget, AnyJev L2 is fit on the labelled bookings (Decider.fit_head, then level="L2");
run_label_curve_tabular.py fits logistic regression and LightGBM on the very same rows. Small budgets are
noisy, so they are repeated on several independent stratified draws from the calibration pool;
repeat 0 uses the exact 300 bookings of the main L1 / L2 / low-data runs. The zero-label point is
AnyJev L0 from run_l0.py. Hidden states are cached per prompt, so every booking goes through the
model once no matter how many heads are fit.
"""
import time

import numpy as np
import pandas as pd

from common import save_json, setup
from label_budgets import draws
from llm_common import CANCEL, load_backend, load_split, make_decider, question, states_and_labels
from metrics import compute

KEYS = ("accuracy", "f1", "roc_auc", "brier", "ece", "coverage_at_5pct_error", "coverage_at_10pct_error")


def main():
    cfg = setup(__doc__)
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    y_test = test["is_canceled"].values
    states, _ = states_and_labels(test)
    fits = draws(cfg)
    need = {b for _, _, rows in fits for b in rows.booking_id}
    print(f"[plan] {len(fits)} L2 heads; {len(need)} distinct labelled bookings + {len(states)} test bookings "
          f"each go through the model once")
    be = load_backend(cfg)
    q = question(cfg)
    rows, t0 = [], time.time()
    for n, r, train in fits:
        c_states, c_labels = states_and_labels(train)
        d = make_decider(be)
        art = d.fit_head(q, c_states, c_labels, seed=r)
        p = np.array([x.probs[CANCEL] for x in d.decide_batch(states, q, level="L2", require="L2")])
        m = compute(y_test, p, cfg)
        rows.append({"method": "anyjev_l2", "n_labels": n, "repeat": r, "n_pos": int(train.is_canceled.sum()),
                     **{k: m[k] for k in KEYS}, "l2_layer": art["layer_abs"]})
        print(f"[curve] n={n:4d} rep={r}  L2 acc {m['accuracy']:.3f}  auc {m['roc_auc']:.3f}  "
              f"(block {art['layer_abs']}, {time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(cfg["paths"]["outputs"] / "label_curve_l2.csv", index=False, float_format="%.4f")
    save_json({"seconds": time.time() - t0, "n_fits": len(fits)}, cfg["paths"]["outputs"] / "run_label_curve.json")


if __name__ == "__main__":
    main()
