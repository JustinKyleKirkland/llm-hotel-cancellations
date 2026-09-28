"""(d) Order-flip test: swap the answer option order ("Answer Yes or No." -> "Answer No or Yes.")
and count how many cancel / keep decisions change.

naive : AnyJev's raw readout under each phrasing (the raw level itself only ever reads the first).
L0    : AnyJev's L0 readout with the option order reversed. For a yes/no question L0 already reads
        both orders and combines them symmetrically, so this is 0 by construction -- verified here by
        recomputing it with the order reversed rather than assumed.
Ablation "prior only": the batch-prior correction applied to one phrasing at a time (no order
averaging), which shows which half of L0 removes the order effect.
All numbers come from one decide_batch(level="L0") call (the forwards are cached from run_l0).
"""
import numpy as np
from anyjev.calibrate import apply_contextual, marginalize

from common import save_json, setup
from llm_common import CANCEL, load_backend, load_split, make_decider, question, states_and_labels


def main():
    cfg = setup(__doc__)
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    states, _ = states_and_labels(test)
    be = load_backend(cfg)
    d = make_decider(be)
    decs = d.decide_batch(states, question(cfg), level="L0")
    strength = d.strength()
    rows = {"naive": ([], []), "prior_only": ([], []), "anyjev_l0": ([], [])}
    max_dev = 0.0
    for dec in decs:
        g = dec.diagnostics
        P, perms, prior = np.asarray(g["p_pos_raw"]), g["perms"], np.power(g["prior"], strength)
        # naive: raw distribution under each phrasing, mapped from positions to options
        for s in (0, 1):
            rows["naive"][s].append(marginalize(P[s:s + 1], perms[s:s + 1])[CANCEL])
            rows["prior_only"][s].append(marginalize(apply_contextual(P[s:s + 1], prior[s:s + 1]),
                                                     perms[s:s + 1])[CANCEL])
        # L0 as listed, and L0 with the two orders swapped
        fwd = marginalize(apply_contextual(P, prior), perms, d.combine)
        rev = marginalize(apply_contextual(P[::-1], prior[::-1]), perms[::-1], d.combine)
        max_dev = max(max_dev, float(abs(fwd[CANCEL] - dec.probs[CANCEL])))
        rows["anyjev_l0"][0].append(fwd[CANCEL])
        rows["anyjev_l0"][1].append(rev[CANCEL])
    assert max_dev < 1e-9, f"recomputed L0 differs from AnyJev's by {max_dev}"

    y = test["is_canceled"].values
    out = {"n": len(decs), "question": cfg["question"], "model": cfg["model"]["name"],
           "l0_recompute_max_abs_diff": max_dev, "methods": {}}
    for name, (a, b) in rows.items():
        a, b = np.asarray(a), np.asarray(b)
        flips = (a >= 0.5) != (b >= 0.5)
        out["methods"][name] = {
            "flip_rate": float(flips.mean()),
            "n_flipped": int(flips.sum()),
            "mean_abs_prob_change": float(np.abs(a - b).mean()),
            "acc_yes_first": float(((a >= 0.5) == y).mean()),
            "acc_no_first": float(((b >= 0.5) == y).mean()),
            "cancel_rate_pred_yes_first": float((a >= 0.5).mean()),
            "cancel_rate_pred_no_first": float((b >= 0.5).mean()),
        }
        print(f"{name:11s} flip rate {flips.mean():.3f}  ({flips.sum()} of {len(a)})  "
              f"mean |dp| {np.abs(a - b).mean():.3f}")
    save_json(out, cfg["paths"]["outputs"] / "order_flip.json")


if __name__ == "__main__":
    main()
