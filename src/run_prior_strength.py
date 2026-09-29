"""(extra) How strong should L0's label-prior correction be? AnyJev divides by the batch prior raised
to `prior_strength` (default 0.75). This reruns L0 at 0, 0.5, 0.75 and 1.0 for the plain question and
the hinted one. Every forward is already cached, so the model is never loaded."""
import numpy as np
import pandas as pd
from anyjev import Decider

from common import save_json, setup
from llm_common import CANCEL, load_backend, load_split, question, states_and_labels
from metrics import compute

STRENGTHS = (0.0, 0.5, 0.75, 1.0)


def main():
    cfg = setup(__doc__)
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    y = test["is_canceled"].values
    no_dep = (test["deposit_type"] == "No Deposit").values
    states, _ = states_and_labels(test)
    be = load_backend(cfg)
    rows = []
    for hint in (False, True):
        q = question(cfg, hint=hint)
        for s in STRENGTHS:
            d = Decider(be, shared_prefix=False, prior_strength=s)
            p = np.array([x.probs[CANCEL] for x in d.decide_batch(states, q, level="L0")])
            m = compute(y, p, cfg)
            sub = compute(y[no_dep], p[no_dep], cfg)
            rows.append({"hint": hint, "prior_strength": s, "accuracy": m["accuracy"], "f1": m["f1"],
                         "roc_auc": m["roc_auc"], "brier": m["brier"], "ece": m["ece"],
                         "pred_cancel_rate": m["pred_cancel_rate"], "mean_p_cancel": float(p.mean()),
                         "coverage_at_5pct_error": m["coverage_at_5pct_error"],
                         "roc_auc_no_deposit": sub["roc_auc"]})
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["outputs"] / "l0_prior_strength.csv", index=False, float_format="%.4f")
    save_json({"rows": rows}, cfg["paths"]["outputs"] / "l0_prior_strength.json")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
