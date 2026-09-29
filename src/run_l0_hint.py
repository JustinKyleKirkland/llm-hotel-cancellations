"""(extra) Zero labels plus one sentence: the same yes/no question with a revenue manager's note in
front of it (config `hint`, taken from the training split only). Scores the naive readout
(level="raw") and AnyJev L0 exactly as run_naive.py / run_l0.py do. No labels are used."""
import time

import numpy as np

from common import save_json, setup
from llm_common import CANCEL, estimate_prompts, load_backend, load_split, make_decider, question, save_preds, states_and_labels


def main():
    cfg = setup(__doc__)
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    states, _ = states_and_labels(test)
    q = question(cfg, hint=True)
    be = load_backend(cfg)
    estimate_prompts(len(states), 2, be)
    t0 = time.time()
    decs = make_decider(be).decide_batch(states, q, level="L0")
    raw = make_decider(be).decide_batch(states, q, level="raw")     # same prompts: served from the cache
    g = [d.diagnostics for d in decs]
    save_preds(cfg, "naive_hint", test, [d.probs[CANCEL] for d in raw])
    save_preds(cfg, "anyjev_l0_hint", test, [d.probs[CANCEL] for d in decs],
               {"order_flip_raw": [x["order_flip_raw"] for x in g]})
    save_json({"method": "anyjev_l0_hint", "question": q.text, "model": cfg["model"]["name"], "n": len(decs),
               "prior_strength": g[-1]["prior_strength"],
               "batch_prior_by_phrasing": np.asarray(g[-1]["prior"]).tolist() if g[-1]["prior"] is not None else None,
               "seconds": time.time() - t0}, cfg["paths"]["outputs"] / "run_l0_hint.json")


if __name__ == "__main__":
    main()
