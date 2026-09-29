# llm-hotel-cancellations

Can a small open LLM tell which hotel bookings will be canceled, and can you trust its
probabilities enough to automate part of that call? This repo tests it on 119,390 real bookings
with [AnyJev](https://github.com/nokia-applied-research/AnyJev), and compares it with LightGBM and
logistic regression. Everything runs on an Apple Silicon Mac.

Short version: with no labels the model ranks bookings backwards, one sentence of domain
knowledge helps a little, a few hundred labels fix it, and logistic regression still does better
at every label budget.

## Data

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*,
22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126 (CC BY 4.0)

`prepare_data.py` looks for the paper's supplementary files (`H1.csv`, `H2.csv`) in `data/raw/`.
If they aren't there, it downloads the
[TidyTuesday copy](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11)
of the same bookings. After cleaning, the two sources are identical cell for cell, so either one
reproduces the results in `outputs/`.

## Running it

```bash
make setup    # python3 -m venv .venv && pip install -r requirements.txt
make smoke    # 50 test bookings, a few minutes; writes to outputs/smoke/ and figures/smoke/
make all      # the full run: 2,000 test bookings, about 2 hours on an M1 Mac mini
```

`./run_all.sh [--smoke]` does the same without make, and every script in `src/` also runs on its
own. Settings (model, sample sizes, seed, the one-sentence hint) live in `config.yaml`.

## What's in `src/`

| script | what it does |
|---|---|
| `prepare_data.py` | cleans the data, drops leaky columns, makes stratified train / calibration / test splits |
| `booking_text.py` | turns a booking row into a short note (examples in `outputs/example_prompts.md`) |
| `run_naive.py` | AnyJev `level="raw"`: one prompt, softmax over the Yes/No tokens |
| `run_l0.py` | AnyJev `level="L0"`: both answer orders combined, label bias divided out, no labels |
| `run_l1.py` | `Decider.calibrate()` on 300 labels (temperature scaling), then `level="L1"` |
| `run_l2.py` | `Decider.fit_head()` on the same 300 labels (a linear head on hidden states), then `level="L2"` |
| `run_l0_hint.py` | raw and L0 with a one-sentence note from the revenue manager in front of the question |
| `run_prior_strength.py` | L0 with its bias correction at strength 0, 0.5, 0.75 and 1.0 |
| `run_order_flip.py` | "Yes or No" vs "No or Yes": how many decisions change, and the ranking under each order |
| `run_classical.py` | LightGBM and logistic regression on the full training split |
| `run_lowdata.py` | the same two models on the 300 bookings L1 and L2 get |
| `run_label_curve.py` | AnyJev L2 at 25, 50, 100, 300 and 1,000 labels |
| `run_label_curve_tabular.py` | the tabular models on exactly the same labelled rows |
| `evaluate.py` | accuracy, F1, ROC-AUC, Brier, ECE, coverage vs accuracy, bootstrap 95% CIs |
| `make_figures.py` | everything in `figures/` |

Results end up in `outputs/` (`metrics.json` has every number) and charts in `figures/`.

## Choices and gotchas

- **Model.** `Qwen/Qwen3-1.7B` in float16 on MPS, because that's what fits in 8 GB. The M1 has no
  native bfloat16, and float16 was faster (0.49 vs 0.68 s per prompt) with near-identical
  outputs. On a bigger machine, change `model.name` in `config.yaml`.
- **Leakage.** `reservation_status` and its date give away the outcome, and
  `assigned_room_type` is set at check-in, so all three are dropped. The dataset's authors note
  that a few other fields can be corrected at check-in, which is a smaller leak that affects
  every method equally.
- **Missing values.** `children` has 4 blanks, set to 0. `NULL` in `agent` and `company` means
  "not applicable", so it stays as its own category. Bookings with zero guests are dropped.
- **Splits.** Stratified on `is_canceled` with seed 42: 70% train, 10% calibration pool, 20% test
  pool. Every method is scored on the same stratified sample of 2,000 test bookings.
- **The hint.** The sentence in `config.yaml` uses only facts from the training split (99.3% of
  its non-refundable bookings canceled, and 97% of those were group or tour-operator bookings).
- **Caching.** `llm_common.CachedHFBackend` wraps AnyJev's `HFBackend` and stores every prompt's
  log-probs and hidden states in `outputs/cache/`, so each booking goes through the model once no
  matter how many experiments reuse it. Delete that folder to start fresh.
- **Memory on 8 GB Macs.** With other apps open, Metal can run out of memory and crash while loading
  the weights. The backend only loads the model when something isn't cached, so rerunning after a
  crash picks up where it stopped. Closing a browser first helps.
- **torch and LightGBM in one process.** Each brings its own OpenMP runtime, and loading both
  segfaults on macOS. That's why the label curve is split into two scripts.
- **Metrics.** ECE uses top-label confidence and 15 equal-width bins. "Auto at 5% error" is the
  largest share of bookings you can decide automatically, most confident first, while keeping the
  error on that share at or below 5%.

## Main results

On 2,000 held-out bookings (ROC-AUC, higher is better, 0.5 is a coin flip):

| method | labels | ROC-AUC |
|---|---|---|
| AnyJev L0 | 0 | 0.388 |
| AnyJev L0 + one-sentence hint | 0 | 0.588 |
| AnyJev L2 | 300 | 0.817 |
| logistic regression | 300 | 0.843 |
| LightGBM | 83,447 | 0.959 |

The two answer orders rank bookings in opposite directions: 0.325 ("Yes or No") vs 0.545
("No or Yes") for the plain question, and 0.709 vs 0.331 with the hint. The full table, with
confidence intervals, is in `outputs/metrics_summary.csv`.
