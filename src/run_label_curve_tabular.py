"""Label-efficiency curve, tabular side: logistic regression and LightGBM (the 300-row settings of
run_lowdata.py) on exactly the labelled rows AnyJev L2 gets in run_label_curve.py."""
import pandas as pd

from classical import fit_lgbm, fit_logreg, to_numpy
from common import setup
from label_budgets import draws
from llm_common import load_split
from metrics import compute
from run_label_curve import KEYS


def main():
    cfg = setup(__doc__)
    test = load_split(cfg, "test", cfg["experiment"]["n_test"])
    rows = []
    for n, r, train in draws(cfg):
        for method, fit in (("logreg", fit_logreg), ("lgbm", fit_lgbm)):
            m = compute(test["is_canceled"].values, to_numpy(fit(train, cfg["seed"], small=True)(test)), cfg)
            rows.append({"method": method, "n_labels": n, "repeat": r, "n_pos": int(train.is_canceled.sum()),
                         **{k: m[k] for k in KEYS}})
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["outputs"] / "label_curve_tabular.csv", index=False, float_format="%.4f")
    print(df.groupby(["method", "n_labels"])[["accuracy", "roc_auc", "ece"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
