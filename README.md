# Beyond Vaccination Coverage: Evaluating Machine Learning for Annual Measles Resurgence Risk Prioritization Across Countries

## Overview

This repository contains Python analysis scripts, experimental outputs, and manuscript materials for a retrospective multi-country study of annual reported measles incidence-surge risk prioritization.

The study evaluates whether machine learning models can improve country-level prioritization compared with a simplified vaccination-coverage shortfall ranking.

## Data sources

The study integrates:

- World Health Organization (WHO) annual reported measles cases
- WHO/UNICEF Estimates of National Immunization Coverage (WUENIC), including MCV1 and MCV2
- World Bank population estimates (SP.POP.TOTL)

The source data cover 2000–2025.

Raw and processed datasets are not currently distributed through this repository. Their redistribution permissions and reproducible preparation procedures require final review.

## Study design

- Training: 2003–2018, 2,144 eligible observations
- Validation: 2019–2021, 396 eligible observations
- Future-year test: 2022–2025, 505 eligible observations
- Unseen-country test: 2022–2025, 119 eligible observations
- Reserved held-out-country history: 600 observations

The primary endpoint is a researcher-defined annual reported-incidence surge, not a WHO-confirmed outbreak.

## Models

The primary models and comparators include:

- Vaccination-coverage shortfall ranking
- Vaccination-only logistic regression
- Combined logistic regression
- Random Forest
- XGBoost

The primary evaluation measure is pooled annual Recall@10%, with selection capacity applied separately within each calendar year.

## Main findings

In the future-year test, Random Forest and XGBoost each detected 43 of 143 surge-positive observations (30.1%) under the annual 10% selection capacity, compared with 19 of 143 (13.3%) using vaccination-coverage shortfall ranking.

The geographically held-out comparison was less certain, and performance for strict new surge onset was limited.

Post-test exploratory incidence-only comparisons suggested limited incremental event-detection benefit from adding national vaccination indicators to the evaluated historical-incidence models.

These findings do not establish prospective outbreak early-warning capability.

## Repository structure

- `src/` — Data processing, model development, evaluation, and exploratory analysis scripts
- `results/` — Recorded experimental results, frozen protocol, and publication outputs
- `figures/` — Manuscript and supplementary figures
- `paper/` — Manuscript, supplementary materials, and protocol documents
- `requirements.txt` — Core Python dependencies
- `requirements_full_lock.txt` — Extended environment dependency lock

## Reproducibility

The original research workflow was implemented using Python 3.12.2.

Core dependencies are listed in `requirements.txt`.

The frozen protocol and associated metadata are provided in:

`results/protocol_freeze_v1/`

The independent-test analysis was previously reproduced in a separate local environment with exact agreement across five checked CSV outputs.

However, the complete raw-data-to-final-results pipeline has not yet been independently reproduced, and raw and processed datasets are not currently included in this repository.

Therefore, this repository should not yet be described as supporting complete end-to-end reproduction from the original source data.

## Exploratory analyses

Post-test exploratory analyses include alternative surge definitions, historical-incidence comparators, historical-median feature ablation, and incidence-only model comparisons.

These analyses were not part of the frozen primary evaluation and should not be interpreted as additional confirmatory independent tests.

## Research limitations

Important limitations include retrospective revisions to vaccination estimates, incomplete surveillance reporting, mathematical dependence between a historical-incidence predictor and the researcher-defined outcome, and limited performance for strict new surge onset.

## Citation

Manuscript citation details will be updated following publication.

## License

No software or data redistribution license has been assigned yet. Licensing will be finalized after reviewing source-data permissions and repository contents.
