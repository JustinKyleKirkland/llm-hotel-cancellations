"""(b) AnyJev L0: both phrasings ("Yes or No" / "No or Yes") combined in log space, then the
label-free batch prior divided out (AnyJev defaults). Zero labels. One decide_batch call over the
whole test set, so the batch prior is estimated from all of it."""
import time

import numpy as np

from common import save_json, setup
from llm_common import CANCEL, estimate_prompts, load_backend, load_split, make_decider, question, save_preds, states_and_labels


def main():
    cfg = setup(__doc__)
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    states, _ = states_and_labels(test)
    be = load_backend(cfg)
    estimate_prompts(len(states), 2, be)
    t0 = time.time()
    decs = make_decider(be).decide_batch(states, question(cfg), level="L0")
    g = [d.diagnostics for d in decs]
    # p_pos_raw[s] is the raw distribution under phrasing s, indexed by position; with perms
    # [[0,1],[1,0]] position 0 is "Yes" in phrasing 0 and "No" in phrasing 1.
    raw_a = [x["p_pos_raw"][0][0] for x in g]
    raw_b = [x["p_pos_raw"][1][1] for x in g]
    save_preds(cfg, "anyjev_l0", test, [d.probs[CANCEL] for d in decs],
               {"p_raw_yes_first": raw_a, "p_raw_no_first": raw_b,
                "order_flip_raw": [x["order_flip_raw"] for x in g],
                "order_flip_l0": [x["order_flip_l0"] for x in g]})
    prior = g[0]["prior"]
    save_json({"method": "anyjev_l0", "level": decs[0].level, "model": cfg["model"]["name"], "n": len(decs),
               "prior_method": g[-1]["prior_method"], "prior_strength": g[-1]["prior_strength"],
               "batch_prior_by_phrasing": None if prior is None else np.asarray(prior).tolist(),
               "mean_answer_mass": float(np.mean([x["answer_mass"] for x in g])),
               "seconds": time.time() - t0}, cfg["paths"]["outputs"] / "run_l0.json")


if __name__ == "__main__":
    main()
