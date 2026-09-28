"""(a) Naive baseline: AnyJev's `raw` level -- one prompt ("Answer Yes or No."), softmax over the
Yes/No tokens, no permutation and no prior correction."""
import time

from common import save_json, setup
from llm_common import CANCEL, estimate_prompts, load_backend, load_split, make_decider, question, save_preds, states_and_labels


def main():
    cfg = setup(__doc__)
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    states, _ = states_and_labels(test)
    be = load_backend(cfg)
    estimate_prompts(len(states), 1, be)
    t0 = time.time()
    decs = make_decider(be).decide_batch(states, question(cfg), level="raw")
    p = [d.probs[CANCEL] for d in decs]
    save_preds(cfg, "naive", test, p, {"answer_mass": [d.diagnostics["answer_mass"] for d in decs]})
    save_json({"method": "naive", "level": decs[0].level, "model": cfg["model"]["name"], "n": len(decs),
               "seconds": time.time() - t0}, cfg["paths"]["outputs"] / "run_naive.json")


if __name__ == "__main__":
    main()
