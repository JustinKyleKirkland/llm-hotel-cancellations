"""(f) Low-data comparison: LightGBM (and logistic regression) trained on only the n_calib labelled
bookings AnyJev L1 gets -- the exact same rows."""
from classical import fit_lgbm, fit_logreg, to_numpy
from common import save_json, setup
from llm_common import load_split, save_preds


def main():
    cfg = setup(__doc__)
    n = cfg["experiment"]["n_calib"]
    small = load_split(cfg, "calib_sample", n)
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    seed = cfg["seed"]
    save_preds(cfg, "lgbm_300", test, to_numpy(fit_lgbm(small, seed, small=True)(test)))
    save_preds(cfg, "logreg_300", test, to_numpy(fit_logreg(small, seed, small=True)(test)))
    save_json({"n_train": len(small), "cancel_rate": float(small.is_canceled.mean())},
              cfg["paths"]["outputs"] / "run_lowdata.json")


if __name__ == "__main__":
    main()
