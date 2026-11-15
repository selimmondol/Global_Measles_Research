#!/usr/bin/env python3
"""Step 14: freeze the *retrospective* Global Measles study protocol BEFORE tests.

Run from Measles project root: python src/14_protocol_freeze.py
Reads ONLY code, input-data bytes, development-only outputs and split metadata.
Does not fit models, evaluate test outcomes, or alter any previous research files.
A pre-existing freeze directory causes a STOP (never silently overwrite a lock).
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'results' / 'protocol_freeze_v1'
DRAFT = ROOT / 'results/decision_reliability/protocol_lock_draft.json'
SPLIT = ROOT / 'results/validation_design/split_assignments.csv'

# Core files are intentionally compulsory. A missing file is a STOP, not a silent skip.
REQUIRED = [
    'data/processed/Measles_Master_Rebuilt.csv',
    'data/processed/Measles_Resurgence_Candidates.csv',
    'data/processed/Measles_Model_Features.csv',
    'results/validation_design/split_assignments.csv',
    'results/validation_design/country_holdout.csv',
    'results/decision_reliability/protocol_lock_draft.json',
    'results/decision_reliability/paired_decision_bootstrap.csv',
    'results/decision_reliability/pooled_decision_summary_checked.csv',
    'results/baseline_models/validation_predictions.csv',
    'results/advanced_ml_models/validation_predictions.csv',
    'results/advanced_ml_models/configuration_manifest.csv',
] + [f'src/{i:02d}_{name}.py' for i,name in [
    (1,'data_preparation'), (2,'data_quality'), (3,'resurgence_definition'),
    (4,'outcome_validation'), (5,'target_robustness'),
    (6,'feature_engineering'), (7,'validation_design'),
    (8,'baseline_models'), (9,'baseline_reliability'),
    (10,'advanced_ml_models'), (11,'advanced_ml_reliability'),
    (12,'decision_evaluation'), (13,'decision_reliability'),
]]

# Hash the original files as immutable evidence when present. If originals are missing,
# the freeze refuses to claim complete source provenance.
RAW = ROOT / 'data/raw'
RAW_HINTS = {
    'WHO measles cases': ('Measles reported cases and incidence', '.xlsx'),
    'WHO vaccination': ('Measles vaccination coverage', '.xlsx'),
    'World Bank population': ('API_SP.POP.TOTL', '.zip'),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def stop(message):
    raise SystemExit('STOP — protocol NOT frozen: ' + message)


def main():
    if DEST.exists():
        stop(f'{DEST} already exists; do not overwrite a previous freeze')
    absent = [x for x in REQUIRED if not (ROOT / x).is_file()]
    if absent:
        stop('Required project files missing:\n  ' + '\n  '.join(absent))

    raw_found = {}
    if not RAW.is_dir():
        stop('data/raw/ missing; original source files must be retained')
    for label, (prefix, suffix) in RAW_HINTS.items():
        candidates = sorted(p for p in RAW.iterdir() if p.is_file()
                            and p.name.startswith(prefix) and p.name.endswith(suffix))
        if len(candidates) != 1:
            stop(f'{label}: expected exactly ONE matching file in data/raw/, found {len(candidates)}. '
                 f'Check names and avoid duplicate original downloads.')
        raw_found[label] = candidates[0]

    draft = json.loads(DRAFT.read_text(encoding='utf-8'))
    expected = {
        'cohort': 'all_eligible_imputed_vaccination',
        'primary_model': 'rf_regularized',
        'secondary_model': 'xgb_regularized',
        'statistical_baseline': 'combined',
        'conventional_comparator': 'coverage_shortfall_rank',
        'primary_operating_capacity_pct': 10,
        'secondary_capacity_pct': 20,
    }
    for key, val in expected.items():
        if draft.get(key) != val:
            stop(f'Draft {key} mismatch. Expected {val!r}; found {draft.get(key)!r}')

    s = pd.read_csv(SPLIT, usecols=['Country','Year','eligible_for_outcome','evaluation_role'])
    if len(s) != 5642 or s.duplicated(['Country','Year']).any():
        stop('Split metadata rows/keys differ from audited study')
    expected_years = {'train':(2003,2018), 'validation':(2019,2021),
                      'test':(2022,2025), 'unseen_country_test':(2022,2025)}
    for name, (lo,hi) in expected_years.items():
        sub = s.loc[s['evaluation_role'].eq(name)]
        if sub.empty or not sub['Year'].between(lo,hi).all():
            stop(f'{name} calendar years invalid')
    dev = set(s.loc[s['evaluation_role'].isin(['train','validation','test']),'Country'])
    unseen = set(s.loc[s['evaluation_role'].eq('unseen_country_test'),'Country'])
    if dev & unseen or len(unseen) != 40:
        stop('Unseen-country holdout integrity failure')
    selected = s.loc[s['eligible_for_outcome'].astype(str).str.lower().eq('true')]
    counts = selected['evaluation_role'].value_counts().to_dict()
    for role, expected_n in {'train':2144,'validation':396,'test':505,
                             'unseen_country_test':119,'reserved_not_used':600}.items():
        if counts.get(role) != expected_n:
            stop(f'{role} eligible count differs from audited count: {counts.get(role)}')

    # Only development-validation prediction files are parsed; future test labels
    # and any future/test prediction file are deliberately NOT opened.
    for rel in ['results/baseline_models/validation_predictions.csv',
                'results/advanced_ml_models/validation_predictions.csv']:
        p = pd.read_csv(ROOT / rel, usecols=['Year'])
        if not set(p['Year'].dropna().astype(int)).issubset({2019,2020,2021}):
            stop(f'Non-validation years detected in {rel}')

    # Freeze exact target, feature definitions and model specifications by hashing
    # their source scripts, plus preserving a human-readable protocol in JSON.
    protocol = {
        'status': 'FROZEN_BEFORE_INDEPENDENT_TEST',
        'study': 'Beyond Vaccination Coverage: Evaluating Machine Learning Early-Warning Systems for Missed Measles Resurgence Across Countries',
        'research_design': 'Retrospective one-year-ahead country-year reported-incidence surge prediction',
        'not_preregistered': True,
        'target': {
            'column': 'surge_C_2x_5pm',
            'operational_label': 'Annual reported measles incidence surge (NOT WHO-confirmed outbreak)',
            'eligibility': 'Current incidence and all 3 consecutive earlier-year incidences observed',
            'definition': 'incidence per million >=5, >=2x prior-3y median, and increase >=0.5 per million',
            'zero_median': 'No division by zero: ratio criterion is incidence >= 2*median; retain zero-median rows in primary analysis',
            'secondary_target': 'Strict new-onset label, analysis contingent on known previous-year negative',
            'sensitivity_targets': ['surge_A_2x_1pm','surge_B_3x_1pm','C excluding zero-median rows','higher incidence thresholds'],
        },
        'cohort': 'all_eligible_imputed_vaccination',
        'train_years': [2003,2018], 'validation_years':[2019,2021],
        'locked_future_years': [2022,2025],
        'held_out_country_count': 40,
        'models': {'primary':'rf_regularized','secondary':'xgb_regularized',
                   'statistical_baseline':'combined','policy_baseline':'coverage_shortfall_rank',
                   'model_configs_source':'src/10_advanced_ml_models.py and results/advanced_ml_models/configuration_manifest.csv',
                   'seed':42},
        'preprocessing': 'Training-only median imputation; training-only scaling for Logistic Regression; no test-driven feature engineering or tuning',
        'features': ['mcv1_lag1','mcv2_lag1','incidence_lag1','incidence_median_3y','incidence_std_3y'],
        'decision_rule': 'For each prediction year, rank eligible observed-outcome country-year rows by descending risk; tie-break ascending country name; select ceil(0.10*n)',
        'primary_metric':'Pooled annual event recall at 10% selection capacity',
        'secondary_capacity':0.20,
        'secondary_metrics':['precision at capacity','false alarms','AUROC','average precision','Brier score','calibration'],
        'comparison_uncertainty':'Country-cluster bootstrap; shared global-year shocks incompletely quantified',
        'critical_limitations': [
            'Eligibility for annual decision ranking is determined retrospectively from observable outcomes; real deployment may not know outcome availability',
            'WHO reported cases are not all true infections',
            'WUENIC latest historical revision is NOT a sequence of real-time prediction vintages',
            'National vaccination coverage is not actual district/local immunity',
            'Country-year records within countries are dependent',
            'Definition C was chosen after exploratory development analysis; this freeze is NOT preregistration',
        ],
        'no_test_tuning':True,
        'draft_fields_at_freeze':draft,
    }
    relative_paths = [ROOT / p for p in REQUIRED] + list(raw_found.values()) + [Path(__file__).resolve()]
    files = {str(p.relative_to(ROOT)): {'sha256':sha256(p),'bytes':p.stat().st_size}
             for p in sorted(set(relative_paths))}
    versions = {'python':platform.python_version()}
    for package in ['numpy','pandas','scikit-learn','xgboost','scipy']:
        try: versions[package] = md.version(package)
        except md.PackageNotFoundError: versions[package] = 'NOT INSTALLED'
    lock = {'created_utc':datetime.now(timezone.utc).isoformat(),
            'protocol':protocol,'software_versions':versions,
            'input_and_code_sha256':files}
    # This directory is created only AFTER all checks and hashing succeed.
    DEST.mkdir(parents=True,exist_ok=False)
    output = DEST / 'frozen_protocol.json'
    output.write_text(json.dumps(lock,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (DEST/'FREEZE_README.txt').write_text(
        'Freeze completed before independent testing. Keep this folder unchanged.\n'
        'Original model scripts and input files are identified by SHA-256 hashes.\n'
        'A SHA-256 hash records file bytes but does NOT make an editable file immutable.\n'
        'Do not rerun Step 14 to replace this snapshot. If a correction is necessary,\n'
        'create a documented amendment and disclose whether any test data were seen.\n'
        'Future and held-out-country outcomes have NOT been evaluated here.\n',encoding='utf-8')
    print('\nSTEP 14: PROTOCOL FREEZE COMPLETE (NO TEST EVALUATION)')
    print('Saved:',output.relative_to(ROOT))
    print('Source/code files hashed:',len(files))
    print('Unseen countries:',len(unseen))
    print('Eligible rows by split:',counts)
    print('Primary model: rf_regularized | primary metric: annual Recall@10%')
    print('Do not edit the frozen protocol or tune against either held-out test set.')

if __name__ == '__main__':
    main()
