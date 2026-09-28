"""(e) Classical baselines on the full training split: LightGBM and logistic regression on the
tabular features (every non-leaky column), plus a LightGBM restricted to the fields the LLM sees.
Scored on the same test sample as the LLM, and on the whole test pool for reference."""
import time

from classical import PROMPT_CATEGORICAL, PROMPT_NUMERIC, fit_lgbm, fit_logreg, to_numpy
from common import save_json, setup
from llm_common import load_split, save_preds
from metrics import compute


def main():
    cfg = setup(__doc__)
    train = load_split(cfg, "train")
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    pool = load_split(cfg, "test_pool")
    seed = cfg["seed"]
    info = {"n_train": len(train), "pool_metrics": {}}
    for name, fit in [("lgbm_full", lambda: fit_lgbm(train, seed)),
                      ("logreg_full", lambda: fit_logreg(train, seed)),
                      ("lgbm_prompt_fields", lambda: fit_lgbm(train, seed, PROMPT_CATEGORICAL, PROMPT_NUMERIC))]:
        t0 = time.time()
        predict = fit()
        save_preds(cfg, name, test, to_numpy(predict(test)))
        m = compute(pool["is_canceled"].values, to_numpy(predict(pool)), cfg)
        info["pool_metrics"][name] = {k: m[k] for k in ("accuracy", "f1", "roc_auc", "brier", "ece")}
        info[f"{name}_seconds"] = time.time() - t0
        print(f"[{name}] whole test pool (n={len(pool)}): " +
              ", ".join(f"{k}={v:.3f}" for k, v in info["pool_metrics"][name].items()))
    save_json(info, cfg["paths"]["outputs"] / "run_classical.json")


if __name__ == "__main__":
    main()
