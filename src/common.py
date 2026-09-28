"""Shared helpers: config loading, paths, the --smoke switch, method names and colors."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# One fixed color per method, used by every chart. Slots 1-6 of the dataviz reference palette in
# their validated (colorblind-checked) order; the naive baseline is a neutral gray.
METHODS = {
    "naive":        {"label": "LLM naive (raw)",        "color": "#8a8984"},
    "anyjev_l0":    {"label": "AnyJev L0 (0 labels)",   "color": "#2a78d6"},
    "lgbm_full":    {"label": "LightGBM (full train)",  "color": "#eb6834"},
    "anyjev_l1":    {"label": "AnyJev L1 (300 labels)", "color": "#1baf7a"},
    "anyjev_l2":    {"label": "AnyJev L2 (300 labels)", "color": "#4a3aa7"},
    "lgbm_300":     {"label": "LightGBM (300 labels)",  "color": "#eda100"},
    "logreg_full":  {"label": "Logistic reg. (full)",   "color": "#e87ba4"},
    "logreg_300":   {"label": "Logistic reg. (300)",    "color": "#008300"},
}


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--smoke", action="store_true", help="tiny run (50 bookings) to check the pipeline")
    return p.parse_args()


def load_config(path: str | Path = ROOT / "config.yaml", smoke: bool = False) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["smoke_mode"] = smoke
    if smoke:
        s = cfg["smoke"]
        cfg["experiment"]["n_test"] = s["n_test"]
        cfg["experiment"]["n_calib"] = s["n_calib"]
        cfg["paths"]["outputs"] = s["outputs"]
        cfg["paths"]["figures"] = s["figures"]
    for key in ("outputs", "figures", "cache"):
        path_ = ROOT / cfg["paths"][key]
        path_.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = path_
    cfg["data"]["processed_dir"] = ROOT / cfg["data"]["processed_dir"]
    return cfg


def setup(description: str) -> dict:
    args = parse_args(description)
    cfg = load_config(args.config, args.smoke)
    seed_everything(cfg["seed"])
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
