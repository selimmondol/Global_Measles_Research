# Beyond Vaccination Coverage: Evaluating Machine Learning for Annual Measles Resurgence Risk Prioritization Across Countries

**Research code, analysis outputs, and supplementary materials** for a retrospective, multi-country study of annual reported-measles-incidence surge prioritization.

**Authors:** Md Selim Mondol¹ · Mst Sumaya Sultana²  
¹ Heinrich Heine University Düsseldorf, Germany  
² Department of Statistics, Begum Rokeya University, Rangpur, Bangladesh  
**Correspondence:** [md.selim.mondol@hhu.de](mailto:md.selim.mondol@hhu.de)

> **Interpretation:** This study evaluates *retrospective ranking* of a researcher-defined annual incidence-surge outcome. It does **not** validate prospective outbreak forecasting, outbreak-onset detection, or an operational early-warning system.

## Overview

We ask whether machine-learning models can prioritize country-years with reported measles-incidence surges more effectively than a simplified vaccination-coverage shortfall ranking, when only the highest-ranked **10% of outcome-evaluable countries per year** can be selected.

The analysis integrates annual WHO-reported measles cases, WHO/UNICEF Estimates of National Immunization Coverage (WUENIC; MCV1 and MCV2), and World Bank population estimates for **2000–2025**. The candidate methods are vaccination-only and combined logistic regression, Random Forest, and XGBoost. Training, validation, future-year testing, and a country-level geographical holdout are separated. Models use lagged information; historical WUENIC estimates may nonetheless reflect later revisions.

## Study design

| Partition | Years | Outcome-eligible country-years |
| --- | --- | ---: |
| Training | 2003–2018 | 2,144 |
| Validation | 2019–2021 | 396 |
| Future-year independent test | 2022–2025 | 505 |
| Geographically held-out independent test | 2022–2025 | 119 |
| Reserved historical observations from held-out countries | 2003–2021 | 600 |

The integrated input panel contains **5,642 country-year records** across **217 countries and territories**; **3,764** met the outcome eligibility criteria. The geographical evaluation involved **40 held-out countries**, of which **36** contributed outcome-evaluable observations during 2022–2025.

The primary outcome requires complete current-year and preceding three-year incidence histories. A surge-positive observation meets three researcher-defined conditions: reported incidence of at least **5 cases per million**, at least **twice** the previous three-year median, and an absolute rise of at least **0.5 cases per million**. This outcome is **not** a WHO-confirmed outbreak definition.

## Principal results

**Annual Recall@10%** is calculated by selecting the top `ceil(0.10 × n)` outcome-evaluable countries **separately for each test year**, then pooling detected positives across years.

| Ranking approach | Future-year test (143 positive observations) | Unseen-country test (33 positive observations) |
| --- | ---: | ---: |
| Vaccination-coverage shortfall | 19/143 (**13.3%**) | 5/33 (**15.2%**) |
| Random Forest | 43/143 (**30.1%**) | 8/33 (**24.2%**) |
| XGBoost | 43/143 (**30.1%**) | 8/33 (**24.2%**) |

Random Forest exceeded the coverage-shortfall comparator by **16.78 percentage points** in the future-year test (95% paired country-cluster bootstrap interval: **8.69–23.27** percentage points; 200 replicates). The unseen-country improvement was **9.09 percentage points**, with an interval including zero. For discrimination, Random Forest achieved future-year **AUROC 0.792** and **average precision 0.639**; the corresponding geographically held-out values were **0.741** and **0.543**.

**Important counter-results and sensitivity findings:**

- **Strict new surge onset:** Future-year Random Forest and coverage ranking each found **7/73 (9.6%)**; in unseen countries, Random Forest found **2/18 (11.1%)** versus **4/18 (22.2%)** for coverage ranking. Thus the primary result does not demonstrate better detection of new surge episodes.
- **Historical-median predictor ablation:** Removing the three-year incidence-median predictor substantially reduced fixed-capacity detection; part of its apparent utility may arise from the mathematical relationship between that predictor and the researcher-defined outcome.
- **Incidence-only exploratory models:** Removing vaccination indicators produced only a small reduction in future-year detection (42/143 versus 43/143 for both tree-based model families). This does **not** imply that vaccination is unimportant to measles epidemiology.
- Probabilities were **not perfectly calibrated**, and all post-test model comparisons are exploratory rather than additional independent confirmatory tests.

The full interpretation, methodology, and caveats are in the manuscript and its supplementary material.

## Data sources and availability

| Source | Measures | Link |
| --- | --- | --- |
| World Health Organization | Annual reported measles cases | [WHO Immunization Data](https://immunizationdata.who.int/global/wiise-detail-page/measles-reported-cases-and-incidence) |
| WHO/UNICEF WUENIC | National MCV1 and MCV2 coverage estimates | [WUENIC information](https://www.who.int/news-room/questions-and-answers/item/who-unicef-estimates-of-national-immunization-coverage) |
| World Bank | Total population (`SP.POP.TOTL`) | [World Bank indicator](https://data.worldbank.org/indicator/SP.POP.TOTL) |

**Raw and processed input datasets are not currently included in this public repository.** In particular, cloning the repository does **not** automatically provide the exact data vintages used in this analysis. Consult the manuscript's data description and repository source-data documentation where available; source-provider download interfaces and retrospectively revised estimates may change. Do not interpret a current re-download as proof of identical historical input files.

The reported country-year predictions are provided as **Supplementary Data 1** where included in the repository/submission materials. The manuscript's data-availability statement remains the authoritative description of what is publicly supplied. Redistribution rights for source datasets should be verified before uploading original or derived inputs.

## Repository guide

| Location | Contents |
| --- | --- |
| [`src/`](src/) | Numbered scripts for preparation, model development, evaluation, and exploratory analyses |
| [`results/`](results/) | Saved outputs, model evaluation, uncertainty results, frozen protocol, and publication tables |
| [`results/protocol_freeze_v1/`](results/protocol_freeze_v1/) | Frozen protocol and checksum records used for independent evaluation |
| [`results/independent_test_v1/`](results/independent_test_v1/) | Recorded independent-test results, where available |
| [`results/publication_results_v4/`](results/publication_results_v4/) | Final publication-oriented figures and tables, where available |
| [`figures/`](figures/) | Figures included with the research materials |
| [`paper/`](paper/) | Manuscript and supplementary documents |
| [`requirements.txt`](requirements.txt) | Core pinned Python dependencies |
| [`requirements_full_lock.txt`](requirements_full_lock.txt) | Extended software environment snapshot |

Older result directories, if present, are retained as development history. **Do not mix outputs from different experiment or publication-result versions** when reconstructing the reported numbers.

## Environment and independent-test reproduction

The documented analysis environment uses **Python 3.12.2**. On macOS or Linux:

```bash
# Clone and enter the repository
git clone https://github.com/selimmondol/Global_Measles_Research.git
cd Global_Measles_Research

# Create an isolated environment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Required before execution:** Obtain the original WHO, WUENIC, and World Bank inputs and the intermediate datasets required by the pipeline; restore their expected names and paths according to the source scripts and local data-provenance documentation. Pay attention to capitalization (`Data/` versus `data/`) on case-sensitive filesystems. `requirements_full_lock.txt` is an additional environment record, not proof that all packages or data files can be reconstructed on every platform.

The protected independent-test script is [`src/15_independent_test_evaluation.py`](src/15_independent_test_evaluation.py). **Do not run it against an existing results directory without preserving that directory.** The script is designed not to overwrite protected independent-test outputs. Reproduction should use a separate clone or working copy, with reference output files retained separately before executing the script.

A prior fresh-clone reproduction, with original research inputs supplied locally, verified the frozen protocol's **28 tracked hashes** and obtained exact agreement for five checked CSV outputs:

```text
test_classification_metrics.csv
annual_decision_metrics.csv
pooled_decision_summary.csv
test_predictions.csv
selected_and_missed_test_rows.csv
```

**Scope of this verification:** It establishes independent-test output agreement **conditional on the supplied original input files**. It does **not** establish that a public clone alone reproduces the full raw-data-to-publication workflow.

## Further analyses and reproducibility notes

The repository records robustness checks for alternative incidence-surge thresholds, strict new onset, completeness of vaccination estimates, historical-incidence ranking comparators, predictor ablation, and incidence-only models. These were conducted after the frozen independent evaluation and are explicitly **exploratory**.

The original manuscript describes paired country-cluster bootstrap intervals based on **200 replicates**. A separate **2,000-replicate sensitivity analysis**, if present in the repository, is an additional robustness check and **does not replace** the manuscript's originally reported interval estimates.

## Limitations

- National annual surveillance reports differ in ascertainment, completeness, and reporting practices.
- The primary surge label is researcher-defined, not independently verified outbreak onset.
- The incidence-median predictor also enters the surge definition, creating a mathematical dependency.
- Retrospectively revised WUENIC values do not establish historically available, real-time predictors.
- The unseen-country test includes relatively few positive observations, and its improvement interval includes zero.
- Results are retrospective; prospective public-health effectiveness and probability calibration remain unestablished.

## Manuscript and citation

**Manuscript:** *Beyond Vaccination Coverage: Evaluating Machine Learning for Annual Measles Resurgence Risk Prioritization Across Countries*  
**Authors:** Md Selim Mondol and Mst Sumaya Sultana.

A peer-reviewed publication citation and DOI **have not been added** here; update this section if the article is accepted and published. Do not cite this repository as a published journal article before publication.

## License and use

No repository-wide software or data redistribution license is currently declared. The WHO, WUENIC, and World Bank source data remain subject to their respective providers' terms. The presence of publicly viewable code should not be construed as an explicit grant of reuse or redistribution rights.

## Contact

For correspondence about this study, contact **Md Selim Mondol** at [selimmondol2@gmail.com](mailto:selimmondol2@gmail.com).
