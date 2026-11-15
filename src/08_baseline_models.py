#!/usr/bin/env python3
"""Step 8: leakage-conscious logistic regression BASELINES (development only).

Run from the Measles project root:
    python src/08_baseline_models.py

Inputs (created in Steps 6 and 7):
    data/processed/Measles_Model_Features.csv
    results/validation_design/split_assignments.csv

Outputs (created without overwriting source datasets):
    results/baseline_models/validation_metrics.csv
    results/baseline_models/validation_predictions.csv
    results/baseline_models/cohort_audit.csv
    results/baseline_models/feature_sets.csv
    results/baseline_models/baseline_notes.txt

IMPORTANT: Only TRAIN and VALIDATION records are used; neither future test
nor unseen-country test is inspected for outcomes, fitted, or evaluated.
Target C is an exploratory, researcher-defined annual reported-incidence SURGE,
not a WHO-confirmed outbreak or a finalized clinical endpoint.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
FEATURE_FILE = ROOT / 'data/processed/Measles_Model_Features.csv'
SPLIT_FILE = ROOT / 'results/validation_design/split_assignments.csv'
OUT = ROOT / 'results/baseline_models'
TARGET = 'surge_C_2x_5pm'
SEED = 42

FEATURE_SETS = {
    'vaccination_only': ['mcv1_lag1', 'mcv2_lag1'],
    'incidence_history': ['incidence_lag1', 'incidence_median_3y', 'incidence_std_3y'],
    'combined': ['mcv1_lag1', 'mcv2_lag1', 'incidence_lag1',
                 'incidence_median_3y', 'incidence_std_3y'],
}


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n < 2 or len(np.unique(y)) < 2:
        raise ValueError('Evaluation cohort must contain both outcome classes')
    k = max(1, int(np.ceil(0.10 * n)))
    # Deterministic tie-breaking by row order, consistent across models.
    top = np.argsort(-p, kind='stable')[:k]
    positives = int(y.sum())
    tp, fp, fn, tn = (None,) * 4
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    return {
        'n': n, 'events': positives, 'event_rate': float(y.mean()),
        'AUROC': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)),
        'brier_score': float(brier_score_loss(y, p)),
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'top10_n': k, 'recall_at_10pct': float(y[top].sum() / positives),
        'precision_at_10pct': float(y[top].mean()),
        'threshold_0p5_TP': int(tp), 'threshold_0p5_FP': int(fp),
        'threshold_0p5_FN': int(fn), 'threshold_0p5_TN': int(tn),
    }


def main():
    for path in (FEATURE_FILE, SPLIT_FILE):
        if not path.exists():
            raise SystemExit(f'Missing input: {path}. Run Steps 6 and 7 first.')
    f = pd.read_csv(FEATURE_FILE, low_memory=False)
    s = pd.read_csv(SPLIT_FILE, low_memory=False)
    cols = ['Country', 'Year', 'outcome_eligible', TARGET]
    needed = set(cols + sum(FEATURE_SETS.values(), []))
    if needed - set(f.columns):
        raise SystemExit(f'Missing feature columns: {sorted(needed - set(f.columns))}')
    if set(['Country', 'Year', 'evaluation_role', 'eligible_for_outcome']) - set(s.columns):
        raise SystemExit('Split assignments lack required columns.')
    if f.duplicated(['Country', 'Year']).any() or s.duplicated(['Country', 'Year']).any():
        raise SystemExit('Duplicate country-year key; stop before fitting.')
    # Restrict the split table BEFORE joining targets, to prevent even accidental
    # model access to outcomes from future-test or held-out countries.
    s = s.loc[s['evaluation_role'].isin(['train', 'validation']),
              ['Country', 'Year', 'evaluation_role', 'eligible_for_outcome']].copy()
    d = s.merge(f, on=['Country', 'Year'], how='left', validate='one_to_one', indicator=True)
    if not d['_merge'].eq('both').all():
        raise SystemExit('Some split keys are missing from the feature dataset.')
    if not d['eligible_for_outcome'].astype('boolean').equals(d['outcome_eligible'].astype('boolean')):
        raise SystemExit('Split outcome eligibility differs from feature table.')
    d = d.loc[d['outcome_eligible'].astype('boolean').fillna(False)].copy()
    for c in set(sum(FEATURE_SETS.values(), [])):
        d[c] = pd.to_numeric(d[c], errors='raise')
        d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    if d[TARGET].isna().any() or not d[TARGET].isin([0, 1]).all():
        raise SystemExit('Eligible rows must have binary outcome values.')
    if d['evaluation_role'].value_counts().to_dict() != {'train': 2144, 'validation': 396}:
        print('NOTE: Cohort sizes differ from earlier Step 7 report; review audit.')

    vaccination_columns = FEATURE_SETS['vaccination_only']
    complete_vax = d[vaccination_columns].notna().all(axis=1)
    # PRIMARY fair comparison: all 3 models train and validate on IDENTICAL
    # country-years with both vaccinations observed (no vaccine imputation).
    # SECONDARY: all eligible development years with train-only median imputation.
    cohorts = {
        'matched_complete_vaccination': d.loc[complete_vax].copy(),
        'all_eligible_imputed_vaccination': d.copy(),
    }
    audit, results, pred_rows, feature_rows = [], [], [], []
    for cohort_name, cohort in cohorts.items():
        train = cohort.loc[cohort.evaluation_role.eq('train')].copy()
        valid = cohort.loc[cohort.evaluation_role.eq('validation')].copy()
        if train.empty or valid.empty:
            raise SystemExit(f'No train/validation rows in {cohort_name}')
        if len(np.unique(train[TARGET])) < 2:
            raise SystemExit(f'Insufficient training classes in {cohort_name}')
        audit.append({'cohort': cohort_name, 'train_n': len(train),
                      'train_events': int(train[TARGET].sum()),
                      'validation_n': len(valid),
                      'validation_events': int(valid[TARGET].sum()),
                      'train_countries': train.Country.nunique(),
                      'validation_countries': valid.Country.nunique(),
                      'vaccination_missing_train': int((~train[vaccination_columns].notna().all(axis=1)).sum()),
                      'vaccination_missing_validation': int((~valid[vaccination_columns].notna().all(axis=1)).sum())})
        for name, features in FEATURE_SETS.items():
            # all preprocessing learns parameters strictly from training rows.
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                ('scaler', StandardScaler()),
                ('logit', LogisticRegression(max_iter=2000, solver='lbfgs',
                                             random_state=SEED)),
            ])
            x_train, x_valid = train[features], valid[features]
            pipe.fit(x_train, train[TARGET].astype(int))
            p = pipe.predict_proba(x_valid)[:, 1]
            m = metrics(valid[TARGET], p)
            results.append({'cohort': cohort_name, 'model': name, **m})
            for (country, year, y), prob in zip(
                valid[['Country', 'Year', TARGET]].itertuples(index=False, name=None), p
            ):
                pred_rows.append({'cohort': cohort_name, 'model': name,
                                  'Country': country, 'Year': year,
                                  'observed_surge_label': int(y),
                                  'predicted_probability': float(prob)})
            for column in features:
                feature_rows.append({'model': name, 'feature': column})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT/'validation_metrics.csv', index=False)
    pd.DataFrame(pred_rows).to_csv(OUT/'validation_predictions.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT/'cohort_audit.csv', index=False)
    pd.DataFrame(feature_rows).drop_duplicates().to_csv(OUT/'feature_sets.csv', index=False)
    (OUT/'baseline_notes.txt').write_text(
        'STEP 8: BASELINE LOGISTIC REGRESSION — DEVELOPMENT-ONLY EVALUATION\n'
        'Target: researcher-defined annual incidence surge C, not WHO outbreak status.\n'
        'The outcome is provisional, so results are exploratory, not a publication finding.\n'
        'Development: 2003–2018 training; 2019–2021 validation.\n'
        'Future test 2022–2025 and unseen-country test NEVER loaded into model data.\n'
        'Primary comparison: matched complete MCV1+MCV2 vaccination records across all models.\n'
        'Secondary comparison: all eligible development records, training-only median imputation.\n'
        'Train-only imputation and standardization; same cohort for all three models.\n'
        'LogisticRegression default regularization C=1, no validation-tuned hyperparameters.\n'
        'Recall@10% selects ceil(10% of validation country-years); repeated countries possible.\n'
        'Threshold 0.5 is descriptive, not a chosen policy operating point.\n'
        'No confidence intervals yet; observations within country are dependent.\n'
        'WUENIC series are current-revision historical estimates, not proven vintage-available forecasts.\n'
        'A historical incidence feature enters the surge outcome definition; interpret predictive comparisons cautiously.\n'
        'No country- or year-based random splitting. No test metrics generated.\n',
        encoding='utf-8'
    )
    print('\nSTEP 8: BASELINE MODEL VALIDATION (NO TEST EVALUATION)\n')
    print(pd.DataFrame(audit).to_string(index=False))
    print('\nValidation metrics:\n')
    print(pd.DataFrame(results)[['cohort','model','n','events','AUROC','average_precision',
                                  'brier_score','recall_at_10pct']].round(4).to_string(index=False))
    print(f'\nReports: {OUT.relative_to(ROOT)}/')
    print('NOTE: Test periods and unseen-country holdout untouched; target remains provisional.')


if __name__ == '__main__':
    main()
