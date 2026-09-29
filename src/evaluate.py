"""Collect every method's predictions, compute all metrics (with bootstrap 95% CIs on the test
sample), and write outputs/metrics.json + outputs/metrics_summary.csv."""
import numpy as np
import pandas as pd

from common import METHODS, load_json, save_json, setup
from metrics import compute

EXTRA = {"lgbm_prompt_fields": "LightGBM (prompt fields only)"}
CI_KEYS = ("accuracy", "f1", "roc_auc", "brier", "ece", "coverage_at_5pct_error")


def bootstrap(y, p, cfg, n_boot: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in CI_KEYS}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        m = compute(y[i], p[i], cfg)
        for k in CI_KEYS:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def main():
    cfg = setup(__doc__)
    out_dir = cfg["paths"]["outputs"]
    flips = load_json(out_dir / "order_flip.json")["methods"] if (out_dir / "order_flip.json").exists() else {}
    results, rows = {}, []
    for method in list(METHODS) + list(EXTRA):
        path = out_dir / f"preds_{method}.csv"
        if not path.exists():
            print(f"[skip] {path.name} missing")
            continue
        df = pd.read_csv(path)
        y, p = df["y"].values.astype(int), df["p_cancel"].values
        m = compute(y, p, cfg)
        m["ci95"] = bootstrap(y, p, cfg, n_boot=100 if cfg["smoke_mode"] else 300, seed=cfg["seed"])
        m["order_flip_rate"] = flips.get(method, {}).get("flip_rate")
        m["label"] = METHODS.get(method, {}).get("label", EXTRA.get(method, method))
        results[method] = m
        row = {"method": method, "label": m["label"], "n": m["n"]}
        row.update({k: m[k] for k in ("accuracy", "f1", "roc_auc", "brier", "ece",
                                      "coverage_at_5pct_error", "coverage_at_10pct_error", "order_flip_rate")})
        for c in m["coverage"]:
            row[f"cov@{c['threshold']:.2f}"] = c["coverage"]
            row[f"acc@{c['threshold']:.2f}"] = c["accuracy"]
        rows.append(row)
    save_json(results, out_dir / "metrics.json")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "metrics_summary.csv", index=False, float_format="%.4f")
    show = ["label", "accuracy", "f1", "roc_auc", "brier", "ece", "coverage_at_5pct_error", "order_flip_rate"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table[show].to_string(index=False))
    label_curve(cfg)


def label_curve(cfg):
    """Merge the two halves of the label-efficiency curve and summarise each (method, budget)."""
    out_dir = cfg["paths"]["outputs"]
    parts = [out_dir / f"label_curve_{s}.csv" for s in ("l2", "tabular")]
    if not all(p.exists() for p in parts):
        return
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df.to_csv(out_dir / "label_curve.csv", index=False, float_format="%.4f")
    keys = ("accuracy", "roc_auc", "ece", "brier", "coverage_at_5pct_error")
    summary = (df.groupby(["method", "n_labels"])
                 .agg(n_repeats=("repeat", "size"),
                      **{f"{k}_{s}": (k, s) for k in keys for s in ("mean", "min", "max")})
                 .reset_index())
    save_json({"summary": summary.to_dict(orient="records")}, out_dir / "label_curve.json")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summary[["method", "n_labels", "n_repeats", "accuracy_mean", "accuracy_min", "accuracy_max",
                       "roc_auc_mean", "ece_mean", "coverage_at_5pct_error_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
