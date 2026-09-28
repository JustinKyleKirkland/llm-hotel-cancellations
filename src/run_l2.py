"""(extra) AnyJev L2: a closed-form head (shrunk LDA / ridge) on the model's hidden state, fit with
Decider.fit_head() on the same n_calib labelled bookings L1 gets. Unlike L1 (a temperature, which
cannot change the ranking), L2 can learn this dataset's own patterns. Not in the original brief;
AnyJev's README recommends it as the level to use once you have 100-300 labels."""
import time

from common import save_json, setup
from llm_common import CANCEL, load_backend, load_split, make_decider, question, save_preds, states_and_labels


def main():
    cfg = setup(__doc__)
    n_cal = cfg["experiment"]["n_calib"]
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    calib = load_split(cfg, "calib_sample", n_cal)
    states, _ = states_and_labels(test)
    c_states, c_labels = states_and_labels(calib)
    be = load_backend(cfg)
    be.ensure_loaded()     # L2 reads hidden states, which the logprob cache does not hold
    sec = be.sec_per_prompt or 0.5
    print(f"[plan] {len(c_states) + len(states)} hidden-state forwards (one per booking); "
          f"estimated runtime {(len(c_states) + len(states)) * sec / 60:.1f} min or less")
    t0 = time.time()
    d = make_decider(be)
    q = question(cfg)
    artifact = d.fit_head(q, c_states, c_labels, seed=cfg["seed"])
    t_fit = time.time() - t0
    print(f"[L2] head fit on {len(c_states)} labels in {t_fit:.0f}s: block {artifact['layer_abs']} of "
          f"{artifact['n_blocks']}, {artifact['method']}")
    decs = []
    step = 64
    for i in range(0, len(states), step):
        decs += d.decide_batch(states[i:i + step], q, level="L2", require="L2")
        print(f"[L2] {len(decs)}/{len(states)} decided ({time.time() - t0:.0f}s)", flush=True)
    save_preds(cfg, "anyjev_l2", test, [x.probs[CANCEL] for x in decs])
    meta = {k: v for k, v in artifact.items() if not isinstance(v, (list, dict)) or k == "cv"}
    save_json({"method": "anyjev_l2", "level": decs[0].level, "model": cfg["model"]["name"], "n": len(decs),
               "n_calib": len(c_states), "artifact_meta": meta,
               "blocks_executed": decs[0].diagnostics.get("blocks_executed"),
               "fit_seconds": t_fit, "seconds": time.time() - t0}, cfg["paths"]["outputs"] / "run_l2.json")


if __name__ == "__main__":
    main()
