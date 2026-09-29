"""Medium-friendly charts from outputs/metrics.json and the prediction files.

Every chart: light surface, recessive grid, one fixed color per method (common.METHODS),
fonts sized to stay legible when Medium shrinks the image to phone width, PNG at 2x."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import METHODS, load_json, setup  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
DPI = 200            # figures are laid out at 100 px/inch, saved at 2x
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 19, "axes.labelsize": 15,
    "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "legend.fontsize": 13.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "text.color": INK,
})


def titled(fig, title, subtitle):
    fig.text(0.02, 0.975, title, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=13.5, color=INK2, va="top")


def legend_handles(methods, present, label_fn):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=METHODS[m]["color"], lw=3, label=label_fn(m)) for m in methods if m in present]


def save(fig, cfg, name):
    path = cfg["paths"]["figures"] / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"[fig] {path}")


def reliability(cfg, metrics, n, model):
    fig, ax = plt.subplots(figsize=(8, 7.2))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.80, bottom=0.11)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.text(0.97, 0.90, "perfectly calibrated", rotation=45, color=MUTED, fontsize=12.5,
            ha="right", va="bottom", rotation_mode="anchor", transform=ax.transData)
    for m in ("naive", "anyjev_l0", "anyjev_l1", "anyjev_l2", "lgbm_full"):
        if m not in metrics:
            continue
        r = metrics[m]["reliability"]
        x, y = [b["mean_pred"] for b in r], [b["observed"] for b in r]
        ece = metrics[m]["ece"]
        ax.plot(x, y, color=METHODS[m]["color"], lw=2.5, marker="o", ms=8, mec=SURFACE, mew=2,
                label=f"{METHODS[m]['label']}  ·  ECE {ece:.3f}", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability of cancellation")
    ax.set_ylabel("Share that actually canceled")
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.legend(handles=legend_handles(["naive", "anyjev_l0", "anyjev_l1", "anyjev_l2", "lgbm_full"], metrics,
                                     lambda m: f"{METHODS[m]['label']}  ·  ECE {metrics[m]['ece']:.3f}"),
              loc="upper left", handlelength=1.4, borderaxespad=0.2)
    titled(fig, "Does a 70% mean 70%?",
           f"Predicted vs actual cancel rate, one dot per tenth of the {n:,} test bookings\n"
           f"(sorted by prediction) · LLM = {model}")
    save(fig, cfg, "reliability_diagram.png")


def coverage(cfg, metrics, preds, n, model):
    fig, ax = plt.subplots(figsize=(8, 8.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.83, bottom=0.26)
    thresholds = cfg["experiment"]["thresholds"]
    lowest = 1.0
    order = ["naive", "anyjev_l0", "anyjev_l1", "anyjev_l2", "lgbm_300", "lgbm_full"]
    # L1 is a temperature on L0, so both rank bookings identically and trace the same curve: draw L0
    # dashed on top of L1 so neither hides the other.
    for m in ["naive", "anyjev_l1", "anyjev_l0", "anyjev_l2", "lgbm_300", "lgbm_full"]:
        if m not in preds:
            continue
        df = preds[m]
        p, y = df.p_cancel.values, df.y.values
        conf = np.maximum(p, 1 - p)
        correct = ((p >= 0.5).astype(int) == y)
        idx = np.argsort(-conf, kind="stable")
        cov = np.arange(1, len(y) + 1) / len(y)
        acc = np.cumsum(correct[idx]) / np.arange(1, len(y) + 1)
        start = max(1, int(0.02 * len(y)))       # skip the noisy first 2%
        ax.plot(cov[start:], acc[start:], color=METHODS[m]["color"], lw=2.5, label=METHODS[m]["label"], zorder=3,
                ls=(0, (3, 2.5)) if m == "anyjev_l0" else "-")
        lowest = min(lowest, acc[max(start, int(0.05 * len(y))):].min())
        pts = [c for c in metrics[m]["coverage"] if c["accuracy"] is not None and c["coverage"] >= 0.02]
        ax.plot([c["coverage"] for c in pts], [c["accuracy"] for c in pts], ls="none", marker="o", ms=8,
                color=METHODS[m]["color"], mec=SURFACE, mew=2, zorder=4)
    ax.axhline(0.95, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(0.99, 0.953, "95% accurate", color=INK2, fontsize=12.5, ha="right", va="bottom")
    ax.set_xlim(0, 1.0)
    lo = min(0.5, np.floor(lowest * 10) / 10)
    ax.set_ylim(lo, 1.005)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    yt = np.round(np.arange(lo, 1.0001, 0.1), 2)
    ax.set_yticks(yt, [f"{v:.0%}" for v in yt])
    ax.set_xlabel("Share of bookings decided automatically (most confident first)")
    ax.set_ylabel("Accuracy on the automated share")
    ax.legend(handles=legend_handles(order, preds, lambda m: METHODS[m]["label"]), loc="upper center",
              bbox_to_anchor=(0.45, -0.13), ncol=2, handlelength=1.4, columnspacing=1.2)
    titled(fig, "How much can you automate?",
           f"Dots mark confidence thresholds {thresholds[0]:.2f} to {thresholds[-1]:.2f} · "
           f"{n:,} test bookings\nL0 (dashed) and L1 share one curve: same ranking · LLM = {model}")
    save(fig, cfg, "coverage_vs_accuracy.png")


def order_flip(cfg, flip, n, model):
    names = [("naive", "LLM naive (raw)", METHODS["naive"]["color"], None),
             ("prior_only", "L0 prior only\n(no order averaging)", METHODS["anyjev_l0"]["color"], "//"),
             ("anyjev_l0", "AnyJev L0", METHODS["anyjev_l0"]["color"], None)]
    fr = [flip["methods"][k]["flip_rate"] for k, *_ in names]
    dp = [flip["methods"][k]["mean_abs_prob_change"] for k, *_ in names]
    counts = [flip["methods"][k]["n_flipped"] for k, *_ in names]
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.6), sharey=True)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.74, bottom=0.14, wspace=0.12)
    ypos = np.arange(len(names))[::-1]
    for ax, vals, xlabel, fmt in [(axes[0], fr, "Decisions that flip", "{:.1%}"),
                                  (axes[1], dp, "Mean change in P(cancel)", "{:.3f}")]:
        for yv, v, (_, _, color, hatch) in zip(ypos, vals, names):
            ax.barh(yv, v, height=0.55, color=SURFACE if hatch else color, edgecolor=color,
                    hatch=hatch, lw=2 if hatch else 0, zorder=3)
        top = max(max(vals), 1e-3)
        for yv, v in zip(ypos, vals):
            ax.text(v + top * 0.04, yv, fmt.format(v), va="center", color=INK, fontsize=13.5)
        ax.set_xlim(0, top * 1.6)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlabel(xlabel, fontsize=13.5)
    axes[0].set_yticks(ypos, [lab for _, lab, *_ in names], fontsize=13.5, color=INK)
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    titled(fig, "Swap the answer order",
           f"“Answer Yes or No” vs “Answer No or Yes” · {n:,} test bookings\n"
           f"{counts[0]:,} naive decisions flipped · {model}")
    save(fig, cfg, "order_flip.png")


def order_ranking(cfg, flip, n, model):
    """The answer order changes the ranking, not just the probabilities: ROC-AUC under each order
    alone and under AnyJev L0, for the plain question and with the one-sentence hint."""
    r = flip["ranking_by_order"]
    rows = [("auc_yes_or_no", "“Answer Yes or No” only", METHODS["naive"]["color"], None),
            ("auc_no_or_yes", "“Answer No or Yes” only", METHODS["naive"]["color"], "//"),
            ("auc_l0", "AnyJev L0 (both orders)", METHODS["anyjev_l0"]["color"], None)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.4), sharey=True)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.70, bottom=0.16, wspace=0.10)
    ypos = np.arange(len(rows))[::-1]
    for ax, key, title in [(axes[0], "plain", "Plain question"), (axes[1], "hint", "+ one-sentence hint")]:
        for yv, (k, _, color, hatch) in zip(ypos, rows):
            v = r[key][k]
            ax.barh(yv, v, height=0.55, color=SURFACE if hatch else color, edgecolor=color,
                    hatch=hatch, lw=2 if hatch else 0, zorder=3)
            ax.text(v + 0.015, yv, f"{v:.3f}", va="center", color=INK, fontsize=13.5, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
        ax.axvline(0.5, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(0.5, ypos[0] + 0.45, "coin flip", ha="center", va="bottom", color=INK2, fontsize=12)
        ax.set_xlim(0, 0.9)
        ax.set_xticks([0, 0.5])
        ax.set_xticklabels(["0", "0.5"])
        ax.grid(False)
        ax.set_title(title, fontsize=14.5, color=INK, loc="left", pad=22)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ypos, [lab for _, lab, *_ in rows], fontsize=13, color=INK)
    fig.text(0.62, 0.035, "ROC-AUC (above 0.5 = ranks cancellers higher)", ha="center", color=INK2, fontsize=13)
    titled(fig, "The answer order flips the ranking",
           f"Same model, same bookings, only the order of “Yes” and “No” changes\n"
           f"{n:,} test bookings · {model}")
    save(fig, cfg, "order_ranking.png")


def summary_table(cfg, metrics, n, model):
    order = [m for m in ["naive", "anyjev_l0", "anyjev_l1", "anyjev_l2", "lgbm_300", "logreg_300", "lgbm_full", "logreg_full"]
             if m in metrics]
    cols = [("accuracy", "Accuracy", True, "{:.3f}"), ("f1", "F1", True, "{:.3f}"),
            ("roc_auc", "ROC-AUC", True, "{:.3f}"), ("brier", "Brier ↓", False, "{:.3f}"),
            ("ece", "ECE ↓", False, "{:.3f}"), ("coverage_at_5pct_error", "Auto at\n≤5% err.", True, "{:.0%}"),
            ("order_flip_rate", "Order\nflip ↓", False, "{:.1%}")]
    labels = {"naive": "LLM naive (raw)", "anyjev_l0": "AnyJev L0", "anyjev_l1": "AnyJev L1",
              "anyjev_l2": "AnyJev L2 (extra)",
              "lgbm_300": "LightGBM · 300 rows", "logreg_300": "Log. reg. · 300 rows",
              "lgbm_full": "LightGBM · full train", "logreg_full": "Log. reg. · full train"}
    labels_n = {"lgbm_full": None, "logreg_full": None}
    fig = plt.figure(figsize=(10, 1.5 + 0.55 * (len(order) + 1.3)))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.98 - 1.05 / fig.get_figheight()])
    ax.axis("off")
    ncol = len(cols) + 1
    widths = [0.25] + [0.75 / len(cols)] * len(cols)
    xs = np.cumsum([0] + widths[:-1])
    nrow = len(order) + 1
    rh = 1.0 / (nrow + 0.3)
    head_h = 1.3 * rh
    best = {}
    for key, _, higher, _ in cols:
        vals = [metrics[m].get(key) for m in order]
        vals = [v for v in vals if v is not None]
        if vals:
            best[key] = max(vals) if higher else min(vals)
    for j, (x, w) in enumerate(zip(xs, widths)):
        head = "Method" if j == 0 else cols[j - 1][1]
        ax.text(x + (0.005 if j == 0 else w / 2), 1 - head_h / 2, head, ha="left" if j == 0 else "center",
                va="center", fontsize=12.5, fontweight="bold", color=INK2)
    ax.plot([0, 1], [1 - head_h, 1 - head_h], color=INK2, lw=1.2)
    for i, m in enumerate(order):
        yc = 1 - head_h - rh * (i + 0.5)
        if m in ("lgbm_full",):
            ax.plot([0, 1], [yc + rh / 2, yc + rh / 2], color=GRID, lw=1.2)
        ax.add_patch(plt.Rectangle((0.005, yc - 0.12 * rh), 0.012, 0.24 * rh, color=METHODS[m]["color"]))
        ax.text(0.025, yc, labels[m], ha="left", va="center", fontsize=13, color=INK)
        for j, (key, _, _, fmt) in enumerate(cols):
            v = metrics[m].get(key)
            txt = "n/a" if v is None else fmt.format(v)
            is_best = v is not None and key in best and np.isclose(v, best[key])
            ax.text(xs[j + 1] + widths[j + 1] / 2, yc, txt, ha="center", va="center", fontsize=13,
                    color=INK, fontweight="bold" if is_best else "normal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    del labels_n, ncol
    titled_fig = fig
    titled_fig.text(0.015, 0.985, "Results on held-out bookings", fontsize=18, fontweight="bold", va="top")
    titled_fig.text(0.015, 0.985 - 0.45 / fig.get_figheight(),
                    f"{n:,} test bookings · LLM = {model} · bold = best in column · "
                    "“Auto at ≤5% err.” = share decidable at ≤5% error", fontsize=11.5, color=INK2, va="top")
    save(fig, cfg, "summary_table.png")


def label_curve(cfg, metrics, n, model, metric="accuracy"):
    """How many labels does it take? One line per method over the label budget, a band for the
    spread over resamples, and the two reference points that frame it: always saying "won't
    cancel" (which is what the zero-label LLM does) and LightGBM on the full training split."""
    df = pd.read_csv(cfg["paths"]["outputs"] / "label_curve.csv")
    fig, ax = plt.subplots(figsize=(8, 8.2))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.83, bottom=0.27)
    series = [("anyjev_l2", "AnyJev L2 (LLM + labels)", METHODS["anyjev_l2"]["color"]),
              ("logreg", "Logistic regression", METHODS["logreg_300"]["color"]),
              ("lgbm", "LightGBM", METHODS["lgbm_300"]["color"])]
    for key, label, color in series:
        g = df[df.method == key].groupby("n_labels")[metric].agg(["mean", "min", "max"]).reset_index()
        ax.fill_between(g.n_labels, g["min"], g["max"], color=color, alpha=0.15, lw=0, zorder=2)
        ax.plot(g.n_labels, g["mean"], color=color, lw=2.5, marker="o", ms=8, mec=SURFACE, mew=2,
                label=label, zorder=3)
    full = metrics["lgbm_full"][metric]
    floor = metrics["anyjev_l0"][metric]
    ax.axhline(full, color=METHODS["lgbm_full"]["color"], lw=1.5, ls=(0, (4, 3)), zorder=1)
    n_train = load_json(cfg["paths"]["outputs"] / "run_classical.json")["n_train"]
    ax.axhline(floor, color=MUTED, lw=1.5, ls=(0, (1, 2.5)), zorder=1)
    ax.set_xscale("log")
    budgets = sorted(df.n_labels.unique())
    ax.set_xticks(budgets, [f"{b:,}" for b in budgets])
    ax.minorticks_off()
    lo = min(floor, df[metric].min()) - 0.04
    ax.set_ylim(np.floor(lo * 20) / 20, min(1.0, full + 0.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Labelled bookings used for training")
    ax.set_ylabel({"accuracy": "Accuracy", "roc_auc": "ROC-AUC"}[metric] + f" on {n:,} held-out bookings")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=3, label=lab) for _, lab, c in series]
    handles += [Line2D([], [], color=METHODS["lgbm_full"]["color"], lw=1.5, ls=(0, (4, 3)),
                       label=f"LightGBM, all {n_train:,} bookings ({full:.3f})"),
                Line2D([], [], color=MUTED, lw=1.5, ls=(0, (1, 2.5)),
                       label=f"AnyJev L0, 0 labels ({floor:.3f}"
                             + (", = always “no”)" if metric == "accuracy" else ")"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.43, -0.12), ncol=2, handlelength=1.6,
              columnspacing=1.0, fontsize=12.5)
    reps = df.groupby("n_labels").repeat.nunique()
    titled(fig, "How many labels does it take?",
           f"Every method trains on the same labelled bookings · shaded = range\n"
           f"over {reps.max() if reps.min() == reps.max() else f'{reps.min()} to {reps.max()}'} random draws "
           f"per budget · LLM = {model}")
    save(fig, cfg, "label_curve.png" if metric == "accuracy" else f"label_curve_{metric}.png")


def main():
    cfg = setup(__doc__)
    out = cfg["paths"]["outputs"]
    metrics = load_json(out / "metrics.json")
    preds = {m: pd.read_csv(out / f"preds_{m}.csv") for m in metrics if (out / f"preds_{m}.csv").exists()}
    n = next(iter(metrics.values()))["n"]
    model = cfg["model"]["name"].split("/")[-1]
    reliability(cfg, metrics, n, model)
    coverage(cfg, metrics, preds, n, model)
    if (out / "order_flip.json").exists():
        flip = load_json(out / "order_flip.json")
        order_flip(cfg, flip, n, model)
        if "ranking_by_order" in flip:
            order_ranking(cfg, flip, n, model)
    summary_table(cfg, metrics, n, model)
    if (out / "label_curve.csv").exists():
        label_curve(cfg, metrics, n, model, "accuracy")
        label_curve(cfg, metrics, n, model, "roc_auc")


if __name__ == "__main__":
    main()
