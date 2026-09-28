"""Download / load the hotel booking data, clean it, and make the stratified splits.

Primary source: H1.csv (resort) and H2.csv (city) from the Data in Brief supplement
(Antonio, de Almeida & Nunes, 2019). Fallback: the TidyTuesday mirror of the same data.

Outputs data/processed/{train,calib,test_pool,test,calib_sample}.csv
"""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import ROOT, save_json, setup

# Known after the fact: ReservationStatus(+Date) encode the label; AssignedRoomType is set at check-in.
LEAKAGE = ["reservation_status", "reservation_status_date", "assigned_room_type"]
TARGET = "is_canceled"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def _snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    s = "".join(out)
    return {"a_d_r": "adr", "adr": "adr"}.get(s, s)


def load_paper(raw_dir) -> pd.DataFrame:
    frames = []
    for fname, hotel in (("H1.csv", "Resort Hotel"), ("H2.csv", "City Hotel")):
        df = pd.read_csv(raw_dir / fname, dtype=str)
        df = df.apply(lambda c: c.str.strip())       # the supplement pads categories with spaces
        df.columns = [_snake(c) for c in df.columns]
        df.insert(0, "hotel", hotel)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"a_d_r": "adr"})


def load_mirror(cfg) -> pd.DataFrame:
    path = ROOT / cfg["data"]["mirror_path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {cfg['data']['mirror_url']}")
        urllib.request.urlretrieve(cfg["data"]["mirror_url"], path)
    return pd.read_csv(path, dtype=str)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in LEAKAGE if c in df.columns])
    num = ["is_canceled", "lead_time", "arrival_date_year", "arrival_date_week_number",
           "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults",
           "children", "babies", "is_repeated_guest", "previous_cancellations",
           "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", "adr",
           "required_car_parking_spaces", "total_of_special_requests"]
    for c in num:
        df[c] = pd.to_numeric(df[c].replace({"NA": np.nan, "NULL": np.nan}), errors="coerce")
    # NAs: `children` has 4 missing -> 0 (the paper: missing data only as NULL categories).
    df["children"] = df["children"].fillna(0)
    # Agent / Company "NULL" means "not applicable" (paper, Sec. 2), so it is a category, not a NA.
    for c in ("agent", "company"):
        df[c] = df[c].fillna("NULL").replace({"": "NULL", "NA": "NULL"})
    df["country"] = df["country"].replace({"NULL": "Unknown", "NA": "Unknown"}).fillna("Unknown")
    df["meal"] = df["meal"].replace({"Undefined": "SC"})     # paper: Undefined/SC = no meal package
    df = df.dropna(subset=num).copy()                          # nothing else is missing; guard only
    # Zero-guest bookings (adults = children = babies = 0) are data-entry artifacts; drop them.
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    # One ADR outlier (5400) and one negative ADR; clip to a plausible range.
    df["adr"] = df["adr"].clip(lower=0, upper=1000)
    for c in num:
        if c != "adr":
            df[c] = df[c].astype(int)
    df = df.reset_index(drop=True)
    df.insert(0, "booking_id", np.arange(len(df)))
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n rows with the same cancel rate as df, in a shuffled order (so any prefix is ~stratified too)."""
    n = min(n, len(df))
    sample, _ = train_test_split(df, train_size=n, stratify=df[TARGET], random_state=seed)
    return sample.sample(frac=1.0, random_state=seed)


def main():
    cfg = setup("Prepare and split the hotel booking data")
    raw_dir = ROOT / cfg["data"]["paper_dir"]
    if cfg["data"]["source"] == "paper" and (raw_dir / "H1.csv").exists():
        df, source = load_paper(raw_dir), "paper supplement (H1.csv + H2.csv)"
    else:
        df, source = load_mirror(cfg), "TidyTuesday mirror"
    n_raw = len(df)
    df = clean(df)
    seed = cfg["seed"]
    d = cfg["data"]
    rest, test_pool = train_test_split(df, test_size=d["test_frac"], stratify=df[TARGET], random_state=seed)
    train, calib = train_test_split(rest, test_size=d["calib_frac"] / (1 - d["test_frac"]),
                                    stratify=rest[TARGET], random_state=seed)
    # The full LLM test set (config n_test) and the labelled budget (config n_calib) are stratified
    # samples; the --smoke subsets are the first rows of these same shuffled samples.
    full_n_test = 2000 if cfg["smoke_mode"] else cfg["experiment"]["n_test"]
    test = stratified_sample(test_pool, full_n_test, seed)
    calib_sample = stratified_sample(calib, 300, seed)

    out = cfg["data"]["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("calib", calib), ("test_pool", test_pool),
                       ("test", test), ("calib_sample", calib_sample)]:
        part.to_csv(out / f"{name}.csv", index=False)
    summary = {"source": source, "rows_raw": n_raw, "rows_clean": len(df),
               "cancel_rate": float(df[TARGET].mean()), "dropped_columns": LEAKAGE,
               "splits": {n: {"rows": len(p), "cancel_rate": round(float(p[TARGET].mean()), 4)}
                          for n, p in [("train", train), ("calib", calib), ("test_pool", test_pool),
                                       ("test_sample", test), ("calib_sample", calib_sample)]},
               "hotels": df["hotel"].value_counts().to_dict()}
    save_json(summary, cfg["paths"]["outputs"] / "data_summary.json")
    print(pd.Series(summary["splits"]).to_string())
    print(f"source: {source}; rows {n_raw} -> {len(df)} after cleaning; cancel rate {summary['cancel_rate']:.3f}")


if __name__ == "__main__":
    main()
