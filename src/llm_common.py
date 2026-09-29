"""AnyJev plumbing shared by the LLM experiments.

`CachedHFBackend` is AnyJev's own transformers backend (anyjev.backends.hf.HFBackend) with a disk
cache in front of `next_token_logprobs`: every (prompt, label tokens) pair is scored by the model
once and reused by the naive / L0 / L1 / order-flip scripts, which ask for overlapping prompts.
All probabilities still come out of AnyJev's `Decider`; the cache only avoids repeated forwards.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from anyjev import Decider, Question
from anyjev.backends.hf import HFBackend
from tqdm import tqdm

from booking_text import describe

CANCEL, KEEP = 0, 1          # option indices of Question.noul: ("Yes", "No") -> Yes = will cancel


def pick_device(requested: str) -> str:
    import torch
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested != "cpu":
        print(f"[device] {requested} unavailable, falling back to cpu")
    return "cpu"


class CachedHFBackend(HFBackend):
    """The model itself is loaded lazily, on the first prompt that is not in the cache, so a fully
    cached step (e.g. the order-flip test) only needs the tokenizer -- and never puts the 3-4 GB
    of weights on an 8 GB machine's GPU for nothing."""

    def __init__(self, model_name: str, cache_dir: Path, **kw):
        from transformers import AutoConfig, AutoTokenizer

        self._init_kw = dict(kw)
        self._loaded = False
        self.name, self.device, self.dtype, self.batch_size = model_name, kw.get("device"), kw.get("dtype"), kw.get("batch_size", 16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        mcfg = AutoConfig.from_pretrained(model_name)      # depth and width without loading weights
        self.n_layers, self.hidden_size = int(mcfg.num_hidden_layers), int(mcfg.hidden_size)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        slug = model_name.replace("/", "__")
        self.cache_path = Path(cache_dir) / f"{slug}.{kw.get('dtype', 'bf16')}.pkl"
        self.cache = pickle.loads(self.cache_path.read_bytes()) if self.cache_path.exists() else {}
        self.sec_per_prompt = self.cache.pop("__sec_per_prompt__", None)
        self.hidden_path = Path(cache_dir) / f"{slug}.{kw.get('dtype', 'bf16')}.hidden.pkl"
        self._hidden = None        # prompt hash -> float16 [len(candidate layers), H], loaded on demand

    def ensure_loaded(self):
        if not self._loaded:
            t0 = time.time()
            super().__init__(self.name, **self._init_kw)
            self._loaded = True
            print(f"[model] weights loaded on {self.device}/{self.dtype} in {time.time() - t0:.0f}s")

    # ---- hidden states (AnyJev L2) ----------------------------------------------------------
    def candidate_layers(self):
        """The blocks Decider.fit_head() chooses from by default: 50/60/70/85/100% of depth."""
        n = int(self.n_layers)
        return sorted({max(1, int(round(f * n))) for f in (0.5, 0.6, 0.7, 0.85, 1.0)})

    def hidden_states_to(self, prompts, layers, token_ids=None, positions=None, max_layer=None, lens_ids=None):
        """AnyJev's block-loop feature extractor with a per-prompt cache. Every uncached prompt is run
        once to full depth and all candidate layers are kept, so fitting heads on many label budgets
        re-runs nothing. Anything the cache cannot answer goes to AnyJev's own implementation."""
        self.ensure_loaded()
        cand = self.candidate_layers()
        n_blocks = int(self.n_layers)
        want = [(n_blocks + 1 + i) if i < 0 else int(i) for i in layers]
        if token_ids is not None or positions is not None or lens_ids is not None or not set(want) <= set(cand):
            return super().hidden_states_to(prompts, layers, token_ids, positions, max_layer, lens_ids)
        if self._hidden is None:
            self._hidden = pickle.loads(self.hidden_path.read_bytes()) if self.hidden_path.exists() else {}
        keys = [hashlib.sha1(p.encode()).hexdigest() for p in prompts]
        todo = sorted({k: i for i, k in enumerate(keys) if k not in self._hidden}.values(), key=lambda i: len(prompts[i]))
        if todo:
            print(f"[hidden] {len(todo)} prompts to run to full depth ({len(prompts) - len(todo)} cached); "
                  f"estimated {len(todo) * (self.sec_per_prompt or 0.5) / 60:.1f} min")
            chunk = self.batch_size * 8
            with tqdm(total=len(todo), unit="prompt", smoothing=0.1) as bar:
                for start in range(0, len(todo), chunk):
                    idx = todo[start:start + chunk]
                    feats = super().hidden_states_to([prompts[i] for i in idx], cand, max_layer=n_blocks)[0]
                    for i, f in zip(idx, feats):
                        self._hidden[keys[i]] = f.astype(np.float16)
                    bar.update(len(idx))
                    if (start // chunk) % 10 == 9:
                        self._save_hidden()
            self._save_hidden()
        pos = [cand.index(layer) for layer in want]
        feats = np.stack([self._hidden[k][pos] for k in keys]).astype(np.float32)
        return feats, [None] * len(prompts), None, None

    def _save_hidden(self):
        tmp = self.hidden_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self._hidden))
        tmp.replace(self.hidden_path)

    def _key(self, prompt: str, ids) -> str:
        return hashlib.sha1((prompt + "\x00" + ",".join(map(str, ids))).encode()).hexdigest()

    def save(self):
        blob = dict(self.cache)
        if self.sec_per_prompt:
            blob["__sec_per_prompt__"] = self.sec_per_prompt
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(blob))
        tmp.replace(self.cache_path)

    def next_token_logprobs(self, prompts, token_ids):
        keys = [self._key(p, ids) for p, ids in zip(prompts, token_ids)]
        todo = sorted({k: i for i, k in enumerate(keys) if k not in self.cache}.values(),
                      key=lambda i: len(prompts[i]))
        if todo:
            self.ensure_loaded()
        if todo:
            print(f"[score] {len(todo)} prompts to run through {self.name} on {self.device} "
                  f"({len(prompts) - len(todo)} cached)")
            if self.sec_per_prompt:
                print(f"[score] estimated runtime: {len(todo) * self.sec_per_prompt / 60:.1f} min "
                      f"(measured {self.sec_per_prompt:.2f} s/prompt on an earlier run)")
            chunk = self.batch_size * 4
            done, t_start = 0, time.time()
            with tqdm(total=len(todo), unit="prompt", smoothing=0.1) as bar:
                for start in range(0, len(todo), chunk):
                    idx = todo[start:start + chunk]
                    lps = super().next_token_logprobs([prompts[i] for i in idx], [token_ids[i] for i in idx])
                    for i, lp in zip(idx, lps):
                        self.cache[keys[i]] = lp
                    done += len(idx)
                    bar.update(len(idx))
                    rate = (time.time() - t_start) / done
                    if start == 0 and not self.sec_per_prompt:
                        bar.write(f"[score] estimated runtime: {len(todo) * rate / 60:.1f} min "
                                  f"({rate:.2f} s/prompt measured on the first {done})")
                    if (start // chunk) % 10 == 9:
                        self.save()
            self.sec_per_prompt = (time.time() - t_start) / done
            self.save()
        return [self.cache[k] for k in keys]


def load_backend(cfg) -> CachedHFBackend:
    m = cfg["model"]
    device = pick_device(m["device"])
    dtype = m["dtype"] if device != "cpu" else "float32"
    t0 = time.time()
    be = CachedHFBackend(m["name"], cfg["paths"]["cache"], device=device, dtype=dtype, batch_size=m["batch_size"])
    print(f"[model] {m['name']} on {device}/{dtype} (weights load on the first uncached prompt), "
          f"{len(be.cache)} cached prompts, tokenizer ready in {time.time() - t0:.0f}s")
    return be


def make_decider(backend) -> Decider:
    # AnyJev defaults (batch prior at strength 0.75, log-mean combination). shared_prefix=False
    # sends every prompt through next_token_logprobs so the cache sees all of them; for a yes/no
    # question with two phrasings AnyJev's "auto" setting would not share prefixes anyway (< 3 perms).
    return Decider(backend, shared_prefix=False)


def question(cfg) -> Question:
    return Question.noul(cfg["question"], name="cancel")


def load_split(cfg, name: str, n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["processed_dir"] / f"{name}.csv")
    return df.head(n) if n else df


def states_and_labels(df: pd.DataFrame):
    states = [describe(r) for _, r in df.iterrows()]
    labels = [CANCEL if y else KEEP for y in df["is_canceled"]]
    return states, labels


def save_preds(cfg, method: str, df: pd.DataFrame, p_cancel, extra: dict | None = None) -> Path:
    out = pd.DataFrame({"booking_id": df["booking_id"].values, "y": df["is_canceled"].values,
                        "p_cancel": np.asarray(p_cancel, dtype=float)})
    for k, v in (extra or {}).items():
        out[k] = v
    path = cfg["paths"]["outputs"] / f"preds_{method}.csv"
    out.to_csv(path, index=False)
    print(f"[save] {path}  (n={len(out)}, mean p_cancel={out.p_cancel.mean():.3f}, "
          f"acc@0.5={((out.p_cancel >= 0.5) == out.y).mean():.3f})")
    return path


def estimate_prompts(n_items: int, per_item: int, backend) -> None:
    if backend.sec_per_prompt:
        print(f"[plan] up to {n_items * per_item} prompts; at {backend.sec_per_prompt:.2f} s/prompt that is "
              f"at most {n_items * per_item * backend.sec_per_prompt / 60:.1f} min (cached prompts are free)")
    else:
        print(f"[plan] up to {n_items * per_item} prompts; a runtime estimate prints after the first batch")
