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
        from transformers import AutoTokenizer

        self._init_kw = dict(kw)
        self._loaded = False
        self.name, self.device, self.dtype, self.batch_size = model_name, kw.get("device"), kw.get("dtype"), kw.get("batch_size", 16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        slug = model_name.replace("/", "__")
        self.cache_path = Path(cache_dir) / f"{slug}.{kw.get('dtype', 'bf16')}.pkl"
        self.cache = pickle.loads(self.cache_path.read_bytes()) if self.cache_path.exists() else {}
        self.sec_per_prompt = self.cache.pop("__sec_per_prompt__", None)

    def ensure_loaded(self):
        if not self._loaded:
            t0 = time.time()
            super().__init__(self.name, **self._init_kw)
            self._loaded = True
            print(f"[model] weights loaded on {self.device}/{self.dtype} in {time.time() - t0:.0f}s")

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
