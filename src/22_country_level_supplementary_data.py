"""Export frozen independent-test country-year predictions as Supplementary Data 1.

Run from Measles project root:
    python src/22_country_level_supplementary_data.py

Uses saved independent-test predictions only. Does NOT fit models or modify
frozen data / predictions. Can also run with an explicit input path:
    python src/22_country_level_supplementary_data.py --input results/independent_test_v1/test_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {"test": (505, 143, 53, 43, 43, 19),
            "unseen_country_test": (119, 33, 13, 8, 8, 5)}
METHODS = {"rf": "rf_regularized", "xgb": "xgb_regularized",
           "coverage": "coverage_shortfall_rank"}
OUTPUT_FIELDS = [
    "evaluation_partition", "Country", "Year", "actual_surge_outcome",
    "rf_predicted_probability", "xgb_predicted_probability",
    "vaccination_coverage_shortfall_score", "rf_rank_in_year",
    "xgb_rank_in_year", "coverage_rank_in_year", "annual_top10_n",
    "rf_top10_selected", "xgb_top10_selected", "coverage_top10_selected",
    "rf_correctly_detected_surge", "xgb_correctly_detected_surge",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=root / "results/independent_test_v1/test_predictions.csv")
    ap.add_argument("--output-dir", type=Path,
                    default=root / "results/supplementary_data")
    args = ap.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Missing frozen predictions: {args.input}")

    with args.input.open(newline="", encoding="utf-8-sig") as fh:
        records = list(csv.DictReader(fh))
    required = {"partition", "Country", "Year", "y", *METHODS.values()}
    if not records or required - set(records[0]):
        raise SystemExit(f"Missing columns: {sorted(required - set(records[0]) if records else required)}")

    groups = defaultdict(list)
    keys = set()
    for rec in records:
        part, country, year = rec["partition"], rec["Country"], int(rec["Year"])
        if part not in EXPECTED or year not in (2022, 2023, 2024, 2025):
            raise SystemExit(f"Unexpected test partition/year: {part}/{year}")
        key = (part, country, year)
        if key in keys:
            raise SystemExit(f"Duplicate country-year record: {key}")
        keys.add(key)
        if rec["y"] not in ("0", "1"):
            raise SystemExit(f"Invalid observed outcome: {key}")
        for method in ("rf_regularized", "xgb_regularized"):
            val = float(rec[method])
            if not (0 <= val <= 1 and math.isfinite(val)):
                raise SystemExit(f"Invalid probability {method}: {key}")
        groups[(part, year)].append(rec)

    output = {}
    for (part, year), subset in groups.items():
        n = len(subset)
        k = math.ceil(0.10 * n)
        for prefix, score_col in METHODS.items():
            # Exact frozen annual ranking: score descending, Country ascending on ties.
            ordered = sorted(subset, key=lambda r: (-float(r[score_col]), r["Country"]))
            for rank, row in enumerate(ordered, start=1):
                key = (part, row["Country"], year)
                out = output.setdefault(key, {
                    "evaluation_partition": "future_year_test" if part == "test" else "unseen_country_test",
                    "Country": row["Country"], "Year": year,
                    "actual_surge_outcome": int(row["y"]),
                    "rf_predicted_probability": float(row["rf_regularized"]),
                    "xgb_predicted_probability": float(row["xgb_regularized"]),
                    "vaccination_coverage_shortfall_score": ("Missing" if row["coverage_shortfall_rank"] in ("-inf", "-Infinity")
                                                             else float(row["coverage_shortfall_rank"])),
                    "annual_top10_n": k,
                })
                out[f"{prefix}_rank_in_year"] = rank
                out[f"{prefix}_top10_selected"] = int(rank <= k)
    for out in output.values():
        out["rf_correctly_detected_surge"] = int(out["actual_surge_outcome"] == 1 and out["rf_top10_selected"] == 1)
        out["xgb_correctly_detected_surge"] = int(out["actual_surge_outcome"] == 1 and out["xgb_top10_selected"] == 1)

    ordered_output = sorted(output.values(), key=lambda r: (r["evaluation_partition"], r["Country"], r["Year"]))
    for part, (expected_n, expected_events, expected_selected, expected_rf, expected_xgb, expected_cov) in EXPECTED.items():
        label = "future_year_test" if part == "test" else "unseen_country_test"
        sub = [r for r in ordered_output if r["evaluation_partition"] == label]
        actual = (len(sub), sum(r["actual_surge_outcome"] for r in sub),
                  sum(r["rf_top10_selected"] for r in sub),
                  sum(r["rf_correctly_detected_surge"] for r in sub),
                  sum(r["xgb_correctly_detected_surge"] for r in sub),
                  sum(r["actual_surge_outcome"] * r["coverage_top10_selected"] for r in sub))
        if actual != (expected_n, expected_events, expected_selected, expected_rf, expected_xgb, expected_cov):
            raise SystemExit(f"STOP: {label} expected {(expected_n, expected_events, expected_selected, expected_rf, expected_xgb, expected_cov)}; got {actual}")
        print(f"{label}: {actual[0]} observations, {actual[1]} events, {actual[2]} selected; RF {actual[3]}, XGB {actual[4]}, coverage {actual[5]} detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "Supplementary_Data_1_Country_Year_Predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_output)
    print(f"Saved: {path}")
    print("Frozen model predictions unchanged; only deterministic selection flags added.")


if __name__ == "__main__":
    main()
