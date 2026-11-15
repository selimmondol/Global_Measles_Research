#!/usr/bin/env python3
"""Experiment B: POST-TEST EXPLORATORY historical-median feature ablation.

Run from the original Measles project root:
    python3 src/25_historical_median_ablation.py

IMPORTANT:
- Do NOT edit any original/frozen script, protocol, or prediction file.
- Fit models using 2003–2018 TRAIN rows only.
- Evaluate fixed ablated models on validation and on the existing independent
  2022–2025 partitions. These tests are POST-TEST EXPLORATORY, NOT independent
  confirmatory model evaluations and NOT a new primary model.
- Uses precisely the pre-existing RF/XGB hyperparameters; no tuning/selection.
- Writes only to results/exploratory/historical_median_ablation/ and will not
  overwrite an existing report.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURE_FILE = ROOT / "data" / "processed" / "Measles_Model_Features.csv"
SPLIT_FILE = ROOT / "results" / "validation_design" / "split_assignments.csv"
ORIGINAL_PRED = ROOT / "results" / "independent_test_v1" / "test_predictions.csv"
OUT = ROOT / "results" / "exploratory" / "historical_median_ablation"
TARGET = "surge_C_2x_5pm"
FEATURES = ["mcv1_lag1", "mcv2_lag1", "incidence_lag1", "incidence_std_3y"]
EXPECTED_N = {"train": 2144, "validation": 396, "test": 505,
              "unseen_country_test": 119, "reserved_not_used": 600}
SEED = 42


def build_models():
    return {
        "rf_no_median": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=250, max_depth=5, min_samples_leaf=10,
                max_features="sqrt", random_state=SEED, n_jobs=1,
            )),
        ]),
        "xgb_no_median": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", XGBClassifier(
                n_estimators=180, max_depth=2, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85,
                min_child_weight=10, reg_lambda=10,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=SEED, n_jobs=1,
            )),
        ]),
    }


def load_data():
    for path in (FEATURE_FILE, SPLIT_FILE, ORIGINAL_PRED):
        if not path.is_file():
            raise FileNotFoundError(f"Required original file not found: {path}")
    assignments = pd.read_csv(SPLIT_FILE, low_memory=False)
    features = pd.read_csv(FEATURE_FILE, low_memory=False)
    needed_split = {"Country", "Year", "evaluation_role", "eligible_for_outcome"}
    needed_features = {"Country", "Year", "outcome_eligible", TARGET, *FEATURES}
    for frame, needed, name in [(assignments, needed_split, "split assignments"),
                                 (features, needed_features, "model features")]:
        missing = needed.difference(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame.duplicated(["Country", "Year"]).any():
            raise ValueError(f"{name}: duplicate country-year keys")
    a = assignments[["Country", "Year", "evaluation_role", "eligible_for_outcome"]].copy()
    f = features[["Country", "Year", "outcome_eligible", TARGET, *FEATURES]].copy()
    d = a.merge(f, on=["Country", "Year"], how="left", validate="one_to_one", indicator=True)
    if not d["_merge"].eq("both").all():
        raise ValueError("Some split rows are absent from the full features file")
    ea = d["eligible_for_outcome"].astype("boolean")
    ef = d["outcome_eligible"].astype("boolean")
    if not ea.equals(ef):
        raise ValueError("Outcome eligibility differs across frozen source files")
    d = d.loc[ea.fillna(False)].copy()
    counts = d["evaluation_role"].value_counts().to_dict()
    for role, n in EXPECTED_N.items():
        if counts.get(role) != n:
            raise ValueError(f"Wrong count for {role}: {counts.get(role)}; expected {n}")
    if not d[TARGET].isin([0, 1]).all():
        raise ValueError("Outcome labels are not valid 0/1 values")
    for feature in FEATURES:
        d[feature] = pd.to_numeric(d[feature], errors="raise").replace(
            [np.inf, -np.inf], np.nan)
    if not d.loc[d.evaluation_role.eq("train"), "Year"].between(2003, 2018).all():
        raise ValueError("Training years differ from frozen design")
    if not d.loc[d.evaluation_role.eq("validation"), "Year"].between(2019, 2021).all():
        raise ValueError("Validation years differ from frozen design")
    for role in ("test", "unseen_country_test"):
        if not d.loc[d.evaluation_role.eq(role), "Year"].between(2022, 2025).all():
            raise ValueError(f"Test years differ from frozen design: {role}")
    return d


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite probabilities")
    return {
        "AUROC": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
    }


def annual_recall(base, score, capacity=0.10):
    detected = selected = 0
    for year, group in base.groupby("Year", sort=True):
        if group["Country"].duplicated().any():
            raise ValueError(f"Duplicate country in {year}")
        k = math.ceil(capacity * len(group))
        ranked = group.sort_values([score, "Country"],
                                   ascending=[False, True], kind="stable")
        detected += int(ranked.head(k)["y"].sum())
        selected += k
    positives = int(base["y"].sum())
    return {"events": positives, "selected": selected, "detected": detected,
            "recall_at_10pct": detected / positives if positives else np.nan}


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUT}. Preserve earlier reports; do not overwrite.")
    d = load_data()
    original = pd.read_csv(ORIGINAL_PRED, low_memory=False)
    required = {"partition", "Country", "Year", "y", "rf_regularized", "xgb_regularized"}
    if not required.issubset(original.columns):
        raise ValueError(f"Original prediction CSV missing {sorted(required - set(original.columns))}")
    if original.duplicated(["partition", "Country", "Year"]).any():
        raise ValueError("Duplicate original test predictions")
    original_counts = original.groupby("partition").size().to_dict()
    if original_counts != {"test": 505, "unseen_country_test": 119}:
        raise ValueError(f"Wrong original frozen test sizes: {original_counts}")

    train = d.loc[d.evaluation_role.eq("train")].sort_values(["Country", "Year"])
    models = build_models()
    for model in models.values():
        model.fit(train[FEATURES], train[TARGET].astype(int))

    summaries, prediction_rows = [], []
    for role in ("validation", "test", "unseen_country_test"):
        cohort = d.loc[d.evaluation_role.eq(role)].sort_values(["Country", "Year"])
        base = cohort[["Country", "Year", TARGET]].rename(columns={TARGET: "y"}).copy()
        base["y"] = base["y"].astype(int)
        for name, model in models.items():
            base[name] = model.predict_proba(cohort[FEATURES])[:, 1]

        if role in ("test", "unseen_country_test"):
            check = original.loc[original.partition.eq(role),
                                 ["Country", "Year", "y", "rf_regularized", "xgb_regularized"]]
            base = base.merge(check, on=["Country", "Year"],
                              how="left", validate="one_to_one", suffixes=("", "_original"), indicator=True)
            if not base["_merge"].eq("both").all() or not base["y"].equals(base["y_original"]):
                raise ValueError(f"Original labels do not match for {role}")
            base = base.drop(columns=["_merge", "y_original"])
            names = ("rf_no_median", "xgb_no_median", "rf_regularized", "xgb_regularized")
        else:
            names = ("rf_no_median", "xgb_no_median")

        for name in names:
            result = {"partition": role, "model": name, "features":
                      "4_no_median" if "no_median" in name else "original_frozen_5",
                      "n": len(base), **metrics(base["y"], base[name]),
                      **annual_recall(base, name)}
            summaries.append(result)

        for row in base[["Country", "Year", "y", "rf_no_median", "xgb_no_median"]].itertuples(index=False):
            prediction_rows.append({"partition": role, "Country": row.Country,
                                    "Year": int(row.Year), "y": int(row.y),
                                    "rf_no_median": float(row.rf_no_median),
                                    "xgb_no_median": float(row.xgb_no_median)})

    # Verify original manuscript primary Recall@10% before saving exploratory reports.
    summary = pd.DataFrame(summaries)
    for role, expected in [("test", 43), ("unseen_country_test", 8)]:
        for name in ("rf_regularized", "xgb_regularized"):
            observed = int(summary.loc[(summary.partition == role) &
                                       (summary.model == name), "detected"].iloc[0])
            if observed != expected:
                raise ValueError(f"Frozen primary output mismatch: {role}/{name}: {observed}")

    OUT.mkdir(parents=True, exist_ok=False)
    summary.to_csv(OUT / "median_ablation_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(OUT / "median_ablation_predictions.csv", index=False)
    (OUT / "README.txt").write_text(
        "POST-TEST EXPLORATORY HISTORICAL-MEDIAN ABLATION.\n"
        "Only 2003-2018 training used for refitting; validation and test partitions were not used to fit.\n"
        "Fixed hyperparameters copied from frozen primary scripts; only incidence_median_3y omitted.\n"
        "The existing 5-feature models are represented by the original frozen test predictions.\n"
        "Do NOT use the test results for new model selection or claim independent confirmation.\n"
        "Median ablation does not remove all mathematical links between incidence history and target.\n",
        encoding="utf-8")
    print("\nEXPERIMENT B — POST-TEST EXPLORATORY ABLATION")
    print(summary.round(4).to_string(index=False))
    print("\nSaved:", OUT)
    print("Frozen protocol and original files unchanged.")


if __name__ == "__main__":
    main()
