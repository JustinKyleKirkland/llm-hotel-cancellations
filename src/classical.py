"""Tabular features and models shared by the classical baselines."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
               "reserved_room_type", "deposit_type", "agent", "company", "customer_type"]
NUMERIC = ["lead_time", "arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month",
           "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
           "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
           "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces",
           "total_of_special_requests"]
# The fields the LLM sees in its text description (src/booking_text.py), for a like-for-like check.
PROMPT_CATEGORICAL = ["hotel", "arrival_date_month", "meal", "market_segment", "distribution_channel",
                      "deposit_type", "customer_type"]
PROMPT_NUMERIC = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
                  "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
                  "booking_changes", "adr", "required_car_parking_spaces", "total_of_special_requests"]


def frame(df: pd.DataFrame, cats, nums, categories=None) -> pd.DataFrame:
    X = df[cats + nums].copy()
    for c in cats:
        X[c] = pd.Categorical(X[c].astype(str), categories=None if categories is None else categories[c])
    return X


def fit_lgbm(train: pd.DataFrame, seed: int, cats=CATEGORICAL, nums=NUMERIC, small: bool = False):
    X = frame(train, cats, nums)
    categories = {c: X[c].cat.categories for c in cats}
    if small:   # 300 rows: shallow trees, small leaves, fewer rounds
        params = dict(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=10,
                      min_data_per_group=10, cat_smooth=10, subsample=0.9, subsample_freq=1, colsample_bytree=0.8)
    else:
        params = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                      subsample=0.9, subsample_freq=1, colsample_bytree=0.8)
    model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **params)
    model.fit(X, train["is_canceled"])
    return lambda df: model.predict_proba(frame(df, cats, nums, categories))[:, 1]


def fit_logreg(train: pd.DataFrame, seed: int, cats=CATEGORICAL, nums=NUMERIC, small: bool = False):
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2 if small else 20)
    pre = ColumnTransformer([("cat", enc, cats), ("num", StandardScaler(), nums)])
    model = make_pipeline(pre, LogisticRegression(max_iter=5000, C=0.3 if small else 1.0, random_state=seed))
    model.fit(train[cats + nums].astype({c: str for c in cats}), train["is_canceled"])
    return lambda df: model.predict_proba(df[cats + nums].astype({c: str for c in cats}))[:, 1]


def to_numpy(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
