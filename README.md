# hotel-anyjev

Can an open LLM, read the AnyJev way, predict hotel booking cancellations, and can you trust its
probabilities enough to automate part of the decision? This repo runs the whole comparison
end to end on an Apple Silicon Mac: AnyJev raw / L0 / L1 against LightGBM and logistic
regression, with calibration, order-flip and "how much can I automate" metrics.

- **AnyJev**: <https://github.com/nokia-applied-research/AnyJev> (`pip install "anyjev[hf]"`, v0.2.0,
  Apache-2.0). It reads a typed question's answer probabilities from one prefill of the model's
  next-token distribution. Nothing is generated.
- **Data**: Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.*
  Data in Brief, 22, 41–49. <https://doi.org/10.1016/j.dib.2018.11.126> (CC BY 4.0).
  The paper PDF is in the parent folder: `../main.pdf`.
  (Not found in `~/Zotero`.)

## Quick start

```bash
make setup            # python3 -m venv .venv && pip install -r requirements.txt
make smoke            # 50 test bookings, a few minutes (outputs/smoke/, figures/smoke/)
make all              # the full run from config.yaml (default 2,000 test bookings)
```

`./run_all.sh [--smoke]` is the same thing without make. Every script also runs on its own, e.g.
`.venv/bin/python src/run_l0.py --smoke`.

The data comes from the paper's supplementary file (`1-s2.0-S2352340918315191-mmc2.zip`,
`H1.csv` = resort hotel, `H2.csv` = city hotel) unzipped into `data/raw/`. If those files are
missing, `prepare_data.py` downloads the TidyTuesday mirror of the same 119,390 bookings
(`data.source: mirror` in `config.yaml` forces it).

## Layout

```
config.yaml              model, sample sizes, seed, paths
run_all.sh / Makefile    one command, in order; --smoke for a 50-booking check
src/
  common.py              config, --smoke, method names + fixed chart colors
  prepare_data.py        load, clean, drop leakage columns, stratified train/calib/test split
  booking_text.py        booking row -> hotel-manager description (+ outputs/example_prompts.md)
  llm_common.py          AnyJev HFBackend + disk cache of forwards, Decider, the yes/no Question
  run_naive.py           (a) AnyJev level="raw"
  run_l0.py              (b) AnyJev level="L0" (zero labels)
  run_l1.py              (c) Decider.calibrate() on 300 labels, then level="L1"
  run_order_flip.py      (d) "Yes or No" vs "No or Yes": how many decisions change
  classical.py           tabular features, LightGBM / logistic regression
  run_classical.py       (e) LightGBM + logistic regression on the full training split
  run_lowdata.py         (f) the same models on the 300 rows L1 gets
  metrics.py             accuracy, F1, ROC-AUC, Brier, ECE, coverage vs accuracy
  evaluate.py            all metrics + bootstrap 95% CIs -> outputs/metrics.json, metrics_summary.csv
  make_figures.py        figures/*.png
outputs/                 preds_<method>.csv, metrics.json, order_flip.json, run_*.json, example_prompts.md
figures/                 reliability_diagram.png, coverage_vs_accuracy.png, order_flip.png, summary_table.png
article/draft.md         the Medium draft
```

## What each method is, precisely

| method | what it is |
|---|---|
| `naive` | AnyJev `level="raw"`: one prompt ending "Answer Yes or No.", softmax over the Yes/No token logits. |
| `anyjev_l0` | AnyJev `level="L0"` with the library defaults: both phrasings ("Yes or No" and "No or Yes") combined in log space, then the batch prior (the model's mean answer over the test batch) divided out at strength 0.75. No labels. |
| `anyjev_l1` | `Decider.calibrate(question, 300 states, 300 labels)` fits a temperature over L0 (with the calibration set's prior frozen into the artifact), then `level="L1"`. |
| `lgbm_full`, `logreg_full` | trained on the full training split (83k bookings), every non-leaky column. |
| `lgbm_prompt_fields` | LightGBM on only the fields the LLM sees in its text (reported in the tables, not the charts). |
| `lgbm_300`, `logreg_300` | trained on the same 300 labelled bookings L1 uses. |

The yes/no question is `Question.noul("Based on this booking, will the guest cancel it before arrival?")`.

## Choices worth knowing about

- **Model**: `Qwen/Qwen3-1.7B` in bfloat16 on MPS. This Mac mini has 8 GB of unified memory, so
  Qwen3-4B (8 GB in bf16) does not fit. Qwen3-1.7B is one of the models in AnyJev's own
  small-model table (`docs/results_small_models.md`). Change `model.name` in `config.yaml` on a bigger machine.
- **Leakage**: `reservation_status` and `reservation_status_date` encode the outcome. I also drop
  `assigned_room_type`, which is set at check-in. The paper (Sec. 2) warns that some attributes
  (e.g. party size) can be corrected at check-in, so their distribution differs between canceled
  and non-canceled bookings. That is a residual, milder leak that affects every method.
- **Missing values**: `children` has 4 NAs, which are set to 0. `agent` / `company` `NULL` means
  "not applicable" (paper, Sec. 2), so it is kept as its own category. Bookings with zero guests
  are dropped, and ADR is clipped to [0, 1000].
- **Splits**: stratified on `is_canceled`, seed 42. 70% train, 10% calibration pool, 20% test pool.
  The LLM scores a stratified 2,000-booking sample of the test pool, and every method is scored on
  those same 2,000. Classical models are also scored on the whole test pool (`outputs/run_classical.json`).
- **Order flip for a yes/no question**: AnyJev's L0 reads both phrasings and combines them
  symmetrically, so its flip rate is zero by construction. `run_order_flip.py` recomputes it
  with the order reversed instead of assuming it. It also reports a "prior only" ablation (the
  prior correction without the order averaging) to show which half of L0 does the work.
- **Caching**: `llm_common.CachedHFBackend` subclasses AnyJev's `HFBackend` and caches the
  log-probs of each (prompt, label tokens) pair in `outputs/cache/`. The naive, L0, L1 and flip
  scripts share prompts, so each prompt is scored by the model only once. Delete the cache to
  force fresh forwards.
- **ECE** is on the top-label confidence, with 15 equal-width bins. The reliability diagram bins
  P(cancel) into 10 bins. "Auto @≤5% err" is the largest share of bookings you can decide
  automatically, most confident first, while keeping the error on that share at or below 5%.

## Runtime

The LLM part costs 2 forwards per test booking plus 2 per calibration booking: about 4,600
prompts for the default run. The scripts print an estimate after the first batch, and later
runs reuse the measured speed (see the log for this machine's number). Everything else takes
under a minute.
