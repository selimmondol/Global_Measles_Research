#!/usr/bin/env python3
"""Post-test EXPLORATORY incidence-only RF/XGBoost benchmark.

Put this new script in Measles/src/ and run from the Measles root:
    python src/26_incidence_only_benchmark.py

Fits on the frozen 2003–2018 training cohort only; uses exactly the existing
frozen RF/XGB configurations, without tuning. Reports validation and test
results as descriptive POST-TEST exploratory analyses; never changes the frozen
protocol, model outputs, or source data. Does not establish prospective validity.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "results/protocol_freeze_v1/frozen_protocol.json"
FEATURE_FILE = ROOT / "data/processed/Measles_Model_Features.csv"
SPLIT_FILE = ROOT / "results/validation_design/split_assignments.csv"
FROZEN_PRED = ROOT / "results/independent_test_v1/test_predictions.csv"
OUT = ROOT / "results/exploratory/incidence_only_benchmark"
TARGET = "surge_C_2x_5pm"
FEATURES = ["incidence_lag1", "incidence_median_3y", "incidence_std_3y"]
EXPECTED = {"train": 2144, "validation": 396, "test": 505,
            "unseen_country_test": 119, "reserved_not_used": 600}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_inputs():
    lock = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol = lock["protocol"]
    assert protocol["status"] == "FROZEN_BEFORE_INDEPENDENT_TEST"
    assert protocol["target"]["column"] == TARGET
    assert protocol["train_years"] == [2003, 2018]
    assert protocol["validation_years"] == [2019, 2021]
    manifest = lock["input_and_code_sha256"]
    assert len(manifest) >= 20
    for rel, info in manifest.items():
        file = ROOT / rel
        if not file.is_file() or sha256(file) != info["sha256"]:
            raise RuntimeError(f"FROZEN FILE missing or changed: {rel}")
    print(f"Frozen manifest verified: {len(manifest)} files")


def load_cohorts():
    a = pd.read_csv(SPLIT_FILE, low_memory=False)
    f = pd.read_csv(FEATURE_FILE, low_memory=False)
    assert not a.duplicated(["Country", "Year"]).any()
    assert not f.duplicated(["Country", "Year"]).any()
    assert len(a) == 5642
    cols = ["Country", "Year", "outcome_eligible", TARGET, *FEATURES]
    missing = set(cols) - set(f.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    d = a[["Country", "Year", "evaluation_role", "eligible_for_outcome"]].merge(
        f[cols], on=["Country", "Year"], how="left", validate="one_to_one", indicator=True)
    if not d["_merge"].eq("both").all():
        raise ValueError("Missing feature rows")
    if not d["eligible_for_outcome"].astype("boolean").equals(d["outcome_eligible"].astype("boolean")):
        raise ValueError("Outcome eligibility mismatch")
    d = d.loc[d["eligible_for_outcome"].astype("boolean").fillna(False)].copy()
    if d["evaluation_role"].value_counts().to_dict() != EXPECTED:
        raise ValueError(f"Unexpected split counts: {d['evaluation_role'].value_counts().to_dict()}")
    if not d[TARGET].isin([0, 1]).all():
        raise ValueError("Nonbinary target")
    for col in FEATURES:
        d[col] = pd.to_numeric(d[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    for role, start, end in [("train",2003,2018),("validation",2019,2021),
                             ("test",2022,2025),("unseen_country_test",2022,2025)]:
        assert d.loc[d.evaluation_role.eq(role), "Year"].between(start, end).all()
    dev = set(d.loc[d.evaluation_role.isin(["train", "validation", "test"]), "Country"])
    unseen = set(d.loc[d.evaluation_role.isin(["reserved_not_used", "unseen_country_test"]), "Country"])
    if dev & unseen:
        raise ValueError("Held-out country overlap")
    return d


def make_models():
    return {
        "rf_incidence_only": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=250, max_depth=5, min_samples_leaf=10,
                max_features="sqrt", random_state=42, n_jobs=1)),
        ]),
        "xgb_incidence_only": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", XGBClassifier(
                n_estimators=180, max_depth=2, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85,
                min_child_weight=10, reg_lambda=10,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=42, n_jobs=1)),
        ]),
    }


def evaluate(df, name):
    y = df["y"].to_numpy(dtype=int)
    p = df[name].to_numpy(dtype=float)
    is_coverage_rule = name == "coverage_shortfall_rank"
    # The original coverage rule places missing vaccination inputs at the
    # lowest ranking priority, represented by -inf or NaN in frozen scores.
    # These are valid ranking sentinels, not calibrated probabilities.
    if is_coverage_rule:
        if np.isposinf(p).any():
            raise ValueError("Unexpected positive infinite coverage score")
    elif not np.isfinite(p).all():
        raise ValueError(f"Non-finite model probabilities: {name}")
    selected, detected = 0, 0
    for _, part in df.groupby("Year", sort=True):
        k = math.ceil(len(part) * 0.10)
        chosen = part.sort_values([name, "Country"], ascending=[False,True],
                                  kind="stable", na_position="last").head(k)
        selected += k
        detected += int(chosen["y"].sum())
    events = int(sum(y))
    if is_coverage_rule:
        # Ranking-only comparator: AUROC/AP omitted here because the sentinel
        # handling is specific to the original evaluation; Brier is undefined
        # for an uncalibrated shortfall score.
        auroc = ap = brier = float("nan")
    else:
        auroc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
    return {"n":len(df), "events":events, "selected":selected, "detected":detected,
            "recall_at_10":detected / events if events else float("nan"),
            "AUROC":auroc, "AP":ap, "Brier":brier}


def main():
    if OUT.exists():
        raise FileExistsError(f"Existing exploratory output will not be overwritten: {OUT}")
    for p in [PROTOCOL, FEATURE_FILE, SPLIT_FILE, FROZEN_PRED]:
        if not p.is_file():
            raise FileNotFoundError(p)
    verify_frozen_inputs()
    d = load_cohorts()
    frozen = pd.read_csv(FROZEN_PRED, low_memory=False)
    required = {"partition","Country","Year","y","rf_regularized", "xgb_regularized", "vaccination_only", "coverage_shortfall_rank"}
    if required - set(frozen):
        raise ValueError(f"Missing frozen prediction fields: {sorted(required-set(frozen))}")
    if frozen.duplicated(["partition", "Country", "Year"]).any():
        raise ValueError("Duplicate frozen prediction rows")
    if frozen.groupby("partition").size().to_dict() != {"test":505,"unseen_country_test":119}:
        raise ValueError("Frozen test partitions mismatch")
    train = d.loc[d.evaluation_role.eq("train")].sort_values(["Country","Year"])
    models = make_models()
    for model in models.values():
        model.fit(train[FEATURES], train[TARGET].astype(int))
    summaries, predictions = [], []
    for role in ["validation", "test", "unseen_country_test"]:
        cohort = d.loc[d.evaluation_role.eq(role)].sort_values(["Country","Year"])
        base = cohort[["Country","Year",TARGET]].rename(columns={TARGET:"y"}).copy()
        base["y"] = base["y"].astype(int)
        for name, model in models.items():
            base[name] = model.predict_proba(cohort[FEATURES])[:,1]
        names = list(models)
        if role != "validation":
            ref = frozen.loc[frozen.partition.eq(role),
                ["Country","Year","y","rf_regularized","xgb_regularized",
                 "vaccination_only","coverage_shortfall_rank"]]
            base = base.merge(ref, on=["Country","Year"], how="left", validate="one_to_one",
                              suffixes=("", "_frozen"), indicator=True)
            if not base["_merge"].eq("both").all() or not base["y"].equals(base["y_frozen"]):
                raise ValueError("Frozen labels/keys mismatch")
            base = base.drop(columns=["y_frozen","_merge"])
            names += ["rf_regularized","xgb_regularized","vaccination_only","coverage_shortfall_rank"]
        for name in names:
            summaries.append({"partition":role,"model":name,
                              "status":"exploratory_new_fit" if name in models else "frozen_original",
                              **evaluate(base,name)})
        base.insert(0,"partition",role)
        predictions.append(base)
    summary = pd.DataFrame(summaries)
    for role, expected in [("test",43),("unseen_country_test",8)]:
        actual = summary.loc[(summary.partition == role) & (summary.model == "rf_regularized"), "detected"].iloc[0]
        if int(actual) != expected:
            raise ValueError(f"Frozen reference detection mismatch for {role}: {actual}")
    OUT.mkdir(parents=True, exist_ok=False)
    summary.to_csv(OUT / "incidence_only_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUT / "incidence_only_predictions.csv", index=False)
    (OUT / "README.txt").write_text(
        "POST-TEST EXPLORATORY INCIDENCE-ONLY BENCHMARK.\n"
        "Train = 2003-2018 only; same RF/XGB hyperparameters, no tuning.\n"
        "No MCV1/MCV2 predictors; historical median remains among the 3 incidence features.\n"
        "Frozen full-feature and vaccination-only predictions reused only for comparison.\n"
        "Do not claim this is confirmatory independent testing, a causal vaccine contribution,\n"
        "or a resolution of mathematical dependence between median and surge definition.\n",
        encoding="utf-8")
    print("\nPOST-TEST EXPLORATORY — INCIDENCE-ONLY COMPARISON")
    print(summary.round(4).to_string(index=False))
    print("\nSaved:", OUT)
    print("Original frozen files and results unchanged.")


if __name__ == "__main__":
    main()
