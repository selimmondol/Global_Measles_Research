#!/usr/bin/env python3
"""Exploratory complete-vaccination-data sensitivity analysis.

Reads frozen country-year predictions. Does not retrain models or modify inputs.
Writes results/exploratory/fair_comparator_summary.csv.
"""
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exploratory"

RANKINGS = {
    "Vaccination coverage shortfall": "coverage_shortfall_rank",
    "Vaccination-only LR": "vaccination_only",
    "Combined LR": "combined",
    "Random Forest": "rf_regularized",
    "XGBoost": "xgb_regularized",
}
EXPECTED = {"test": (505, 143), "unseen_country_test": (119, 33)}


def find_predictions():
    candidates = [
        ROOT / "test_predictions.csv",
        ROOT / "results" / "test_predictions.csv",
        ROOT / "results" / "independent_test" / "test_predictions.csv",
        ROOT / "results" / "independent_test_evaluation" / "test_predictions.csv",
        ROOT / "results" / "supplementary_data" / "test_predictions.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted((ROOT / "results").rglob("test_predictions.csv")) if (ROOT / "results").exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError("Multiple test_predictions.csv files found. Choose the frozen original:\n" + "\n".join(map(str, matches)))
    raise FileNotFoundError("test_predictions.csv not found under Measles/ or Measles/results/. Put the original CSV at Measles/test_predictions.csv (without altering it).")


def evaluate(df, capacity=0.10):
    records = []
    for method, col in RANKINGS.items():
        selected_total = 0
        detected_total = 0
        for year, group in df.groupby("Year", sort=True):
            k = math.ceil(len(group) * capacity)
            ranked = group.sort_values(
                [col, "Country"],
                ascending=[False, True],
                kind="stable",
                na_position="last",
            )
            selected = ranked.head(k)
            selected_total += k
            detected_total += int(selected["y"].sum())
        events = int(df["y"].sum())
        records.append({
            "method": method,
            "n": len(df),
            "events": events,
            "selected": selected_total,
            "detected": detected_total,
            "recall_10": detected_total / events if events else float("nan"),
        })
    return pd.DataFrame(records)


def main():
    source = find_predictions()
    print("Reading:", source)
    df = pd.read_csv(source)
    required = {"partition", "Country", "Year", "y", "mcv1_lag1", "mcv2_lag1", *RANKINGS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Required CSV columns missing: " + ", ".join(sorted(missing)))
    if set(df["partition"]) != set(EXPECTED):
        raise ValueError("Unexpected partition names. Expected test and unseen_country_test.")
    if df.duplicated(["partition", "Country", "Year"]).any():
        raise ValueError("Duplicate partition-country-year rows")
    if not df["y"].isin([0, 1]).all():
        raise ValueError("Outcomes must be 0/1")

    outputs = []
    for partition, (expected_n, expected_events) in EXPECTED.items():
        original = df.loc[df["partition"].eq(partition)].copy()
        if (len(original), int(original["y"].sum())) != (expected_n, expected_events):
            raise ValueError(f"Frozen dataset check failed for {partition}")
        complete = original.dropna(subset=["mcv1_lag1", "mcv2_lag1"]).copy()
        for cohort, subset in [
            ("original_all_eligible", original),
            ("complete_mcv1_mcv2", complete),
        ]:
            result = evaluate(subset)
            result.insert(0, "cohort", cohort)
            result.insert(0, "partition", partition)
            outputs.append(result)

    summary = pd.concat(outputs, ignore_index=True)
    expected_detections = {
        "test": {"Vaccination coverage shortfall": 19, "Random Forest": 43, "XGBoost": 43},
        "unseen_country_test": {"Vaccination coverage shortfall": 5, "Random Forest": 8, "XGBoost": 8},
    }
    for partition, rankings in expected_detections.items():
        original = summary.loc[(summary["partition"] == partition) & (summary["cohort"] == "original_all_eligible")]
        for method, expected in rankings.items():
            actual = int(original.loc[original["method"].eq(method), "detected"].iloc[0])
            if actual != expected:
                raise ValueError(f"Frozen-result mismatch: {partition}, {method}: {actual} vs {expected}")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "fair_comparator_summary.csv"
    summary.to_csv(dest, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:", dest)
    print("Original predictions and protocol were not modified.")


if __name__ == "__main__":
    main()
