"""(c) AnyJev L1: temperature scaling on top of L0, fit with Decider.calibrate() on n_calib labelled
bookings from the calibration split (the prior used for the calibration set is frozen into the artifact)."""
import time

from common import save_json, setup
from llm_common import CANCEL, estimate_prompts, load_backend, load_split, make_decider, question, save_preds, states_and_labels


def main():
    cfg = setup(__doc__)
    n_cal = cfg["experiment"]["n_calib"]
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    calib = load_split(cfg, "calib_sample", n_cal)
    states, _ = states_and_labels(test)
    c_states, c_labels = states_and_labels(calib)
    be = load_backend(cfg)
    estimate_prompts(len(states) + len(c_states), 2, be)
    t0 = time.time()
    d = make_decider(be)
    q = question(cfg)
    artifact = d.calibrate(q, c_states, c_labels)            # level="L1": temperature over L0
    decs = d.decide_batch(states, q, level="L1", require="L1")
    save_preds(cfg, "anyjev_l1", test, [x.probs[CANCEL] for x in decs])
    save_json(artifact, cfg["paths"]["outputs"] / "anyjev_l1_artifact.json")
    save_json({"method": "anyjev_l1", "level": decs[0].level, "model": cfg["model"]["name"], "n": len(decs),
               "n_calib": len(c_states), "calib_cancel_rate": sum(y == CANCEL for y in c_labels) / len(c_labels),
               "temperature": artifact["temperature"], "seconds": time.time() - t0},
              cfg["paths"]["outputs"] / "run_l1.json")
    print(f"[L1] temperature = {artifact['temperature']:.3f} fit on {len(c_states)} labelled bookings")


if __name__ == "__main__":
    main()
