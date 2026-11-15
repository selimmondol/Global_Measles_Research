#!/usr/bin/env python3
"""Step 10 — development-only Random Forest and XGBoost model comparison.

Run from Measles project root:
    python src/10_advanced_ml_models.py

Prerequisite:
    python -m pip install xgboost

Inputs:
    data/processed/Measles_Model_Features.csv
    results/validation_design/split_assignments.csv
    results/baseline_models/validation_predictions.csv  (optional cross-check)

Outputs (only in results/advanced_ml_models/):
    validation_metrics.csv, validation_predictions.csv, cohort_audit.csv,
    baseline_comparison.csv, configuration_manifest.csv, advanced_ml_notes.txt

Neither final future test nor held-out country rows are merged, loaded for
model development or evaluated. Target remains PROVISIONAL and does not
correspond to WHO-defined confirmed outbreaks.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit('xgboost is missing. Install in the active environment: python -m pip install xgboost')

ROOT = Path(__file__).resolve().parent.parent
FEATURE_FILE = ROOT / 'data/processed/Measles_Model_Features.csv'
SPLIT_FILE = ROOT / 'results/validation_design/split_assignments.csv'
BASELINE_PRED = ROOT / 'results/baseline_models/validation_predictions.csv'
OUT = ROOT / 'results/advanced_ml_models'
TARGET = 'surge_C_2x_5pm'
FEATURES = ['mcv1_lag1', 'mcv2_lag1', 'incidence_lag1',
            'incidence_median_3y', 'incidence_std_3y']
VAX = ['mcv1_lag1', 'mcv2_lag1']
SEED = 42

# Predeclared small comparison; validation is used to compare configurations,
# not to assert unbiased performance of a selected winner.
CONFIGS = [
    ('random_forest', 'rf_regularized', dict(n_estimators=250, min_samples_leaf=10,
                                            max_features='sqrt', max_depth=5)),
    ('random_forest', 'rf_shallow', dict(n_estimators=250, min_samples_leaf=20,
                                        max_features='sqrt', max_depth=3)),
    ('xgboost', 'xgb_regularized', dict(n_estimators=180, max_depth=2,
                                       learning_rate=0.04, subsample=0.85,
                                       colsample_bytree=0.85, min_child_weight=10,
                                       reg_lambda=10)),
    ('xgboost', 'xgb_shallow', dict(n_estimators=100, max_depth=2,
                                   learning_rate=0.05, subsample=1.0,
                                   colsample_bytree=1.0, min_child_weight=15,
                                   reg_lambda=20)),
]


def evaluate(y, proba):
    y = np.asarray(y, dtype=int)
    p = np.asarray(proba, dtype=float)
    if len(np.unique(y)) != 2 or not np.isfinite(p).all():
        raise ValueError('Need both classes and finite predictions')
    k = max(1, int(np.ceil(0.1 * len(y))))
    top = np.argsort(-p, kind='stable')[:k]
    return dict(n=len(y), events=int(y.sum()), AUROC=roc_auc_score(y,p),
                average_precision=average_precision_score(y,p),
                brier_score=brier_score_loss(y,p),
                recall_at_10pct=float(y[top].sum()/y.sum()),
                precision_at_10pct=float(y[top].mean()), top10_n=k)


def make_model(family, params):
    if family == 'random_forest':
        return RandomForestClassifier(**params, random_state=SEED, n_jobs=1)
    return XGBClassifier(**params, objective='binary:logistic',
                         eval_metric='logloss', random_state=SEED,
                         tree_method='hist', n_jobs=1)


def main():
    for path in (FEATURE_FILE, SPLIT_FILE):
        if not path.exists():
            raise SystemExit(f'Missing {path}. Run Steps 6–7 first.')
    # CRITICAL: Read the partition assignment first and drop all nondevelopment
    # keys BEFORE merging any labels or predictors.
    assignments = pd.read_csv(SPLIT_FILE)
    needed_split = {'Country','Year','evaluation_role','eligible_for_outcome'}
    if not needed_split.issubset(assignments.columns):
        raise SystemExit('Split file is missing required fields')
    if assignments.duplicated(['Country','Year']).any():
        raise SystemExit('Duplicate split keys')
    assignments = assignments.loc[
        assignments['evaluation_role'].isin(['train','validation']),
        ['Country','Year','evaluation_role','eligible_for_outcome']
    ].copy()
    if assignments.empty:
        raise SystemExit('No development rows')
    features = pd.read_csv(FEATURE_FILE, low_memory=False)
    expected = {'Country','Year','outcome_eligible',TARGET,*FEATURES}
    if not expected.issubset(features.columns):
        raise SystemExit(f'Missing columns: {sorted(expected - set(features.columns))}')
    if features.duplicated(['Country','Year']).any():
        raise SystemExit('Duplicate feature keys')
    d = assignments.merge(features[['Country','Year','outcome_eligible',TARGET,*FEATURES]],
                          on=['Country','Year'], how='left', validate='one_to_one', indicator=True)
    if not d['_merge'].eq('both').all():
        raise SystemExit('Unmatched split rows')
    a = d['eligible_for_outcome'].astype('boolean')
    b = d['outcome_eligible'].astype('boolean')
    if not a.equals(b):
        raise SystemExit('Eligibility disagrees between sources')
    d = d.loc[b.fillna(False)].copy()
    if d[TARGET].isna().any() or not d[TARGET].isin([0, 1]).all():
        raise SystemExit('Outcome must be 0/1 for eligible observations')
    for feat in FEATURES:
        d[feat] = pd.to_numeric(d[feat], errors='raise').replace([np.inf,-np.inf], np.nan)
    # Year and country boundaries are asserted; changing the published plan stops execution.
    if not (d.loc[d.evaluation_role.eq('train'),'Year'].between(2003,2018).all()
            and d.loc[d.evaluation_role.eq('validation'),'Year'].between(2019,2021).all()):
        raise SystemExit('Unexpected development calendar years; stop')
    complete = d[VAX].notna().all(axis=1)
    cohorts = {
        'matched_complete_vaccination':d.loc[complete].copy(),
        'all_eligible_imputed_vaccination':d.copy(),
    }
    all_metrics, all_predictions, audits, manifest = [], [], [], []
    for cohort, data in cohorts.items():
        train = data.loc[data.evaluation_role.eq('train')].sort_values(['Country','Year']).copy()
        val = data.loc[data.evaluation_role.eq('validation')].sort_values(['Country','Year']).copy()
        if train.empty or val.empty or train[TARGET].nunique() < 2 or val[TARGET].nunique() < 2:
            raise SystemExit(f'Invalid data for {cohort}')
        if cohort == 'matched_complete_vaccination' and (train[VAX].isna().any().any() or val[VAX].isna().any().any()):
            raise SystemExit('Complete case cohort has missing vaccination features')
        audits.append(dict(cohort=cohort, train_n=len(train), train_events=int(train[TARGET].sum()),
                           validation_n=len(val), validation_events=int(val[TARGET].sum()),
                           missing_vaccination_train=int(train[VAX].isna().any(axis=1).sum()),
                           missing_vaccination_validation=int(val[VAX].isna().any(axis=1).sum())))
        for family, model_name, cfg in CONFIGS:
            # Median imputer is fitted on TRAIN only for every model.
            model = Pipeline([
                ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                ('classifier',make_model(family,cfg))])
            model.fit(train[FEATURES],train[TARGET].astype(int))
            pred = model.predict_proba(val[FEATURES])[:,1]
            score = evaluate(val[TARGET],pred)
            all_metrics.append(dict(cohort=cohort,model=model_name,family=family,**score))
            manifest.append(dict(cohort=cohort,model=model_name,family=family,
                                 features=';'.join(FEATURES),parameters=str(cfg),
                                 random_seed=SEED,training_only_preprocessing=True))
            for (ct,yr,y),p in zip(val[['Country','Year',TARGET]].itertuples(index=False,name=None),pred):
                all_predictions.append(dict(cohort=cohort,model=model_name,
                                            Country=ct,Year=yr,observed_surge_label=int(y),
                                            predicted_probability=float(p)))
    metrics_df=pd.DataFrame(all_metrics)
    pred_df=pd.DataFrame(all_predictions)
    comparison=[]
    if BASELINE_PRED.exists():
        old=pd.read_csv(BASELINE_PRED)
        required={'cohort','model','Country','Year','observed_surge_label','predicted_probability'}
        if required.issubset(old.columns):
            for cohort in cohorts:
                for model_name in ['vaccination_only','incidence_history','combined']:
                    baseline=old.loc[(old.cohort==cohort)&(old.model==model_name)]
                    cohort_predictions=pred_df.loc[pred_df.cohort==cohort]
                    # Align and check labels and country-year keys before comparing.
                    one=cohort_predictions.loc[cohort_predictions.model==CONFIGS[0][1],
                                               ['Country','Year','observed_surge_label']]
                    check=one.merge(baseline[['Country','Year','observed_surge_label','predicted_probability']],
                                    on=['Country','Year'],how='outer',validate='one_to_one',indicator=True,
                                    suffixes=('_ml','_baseline'))
                    if not check['_merge'].eq('both').all() or not check.observed_surge_label_ml.equals(check.observed_surge_label_baseline):
                        raise SystemExit(f'Baseline comparison keys/labels differ: {cohort}/{model_name}')
                    metric=evaluate(check.observed_surge_label_ml,check.predicted_probability)
                    comparison.append(dict(cohort=cohort,model=model_name,family='logistic_baseline',**metric))
    OUT.mkdir(parents=True,exist_ok=True)
    metrics_df.to_csv(OUT/'validation_metrics.csv',index=False)
    pred_df.to_csv(OUT/'validation_predictions.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'cohort_audit.csv',index=False)
    pd.DataFrame(manifest).to_csv(OUT/'configuration_manifest.csv',index=False)
    pd.DataFrame(comparison).to_csv(OUT/'baseline_comparison.csv',index=False)
    (OUT/'advanced_ml_notes.txt').write_text(
        'STEP 10: DEVELOPMENT-ONLY ADVANCED ML COMPARISON\n'
        'Primary target is provisional research-defined country-year reported-incidence surge C, NOT a WHO outbreak.\n'
        'Train 2003–2018; validation 2019–2021. Test 2022–2025 and unseen countries NOT used.\n'
        'Two prespecified cohorts: complete observed MCV1/MCV2 and all eligible with train-only median imputation.\n'
        'Random Forest and XGBoost have a SMALL predeclared configuration list: do not selectively report the winner.\n'
        'All four configurations are reported. If selecting one based on validation, its validation score is selection-biased.\n'
        'All model fitting and preprocessing use only training rows. No global scaling or imputation.\n'
        'The small fixed validation cohort is reused across baseline and ML comparisons: beware repeated-tuning optimism.\n'
        'Recall@10% ranks country-YEAR rows, not distinct countries, with repeated-country selection possible.\n'
        'No temporal or group-bootstrap confidence intervals are computed in this script.\n'
        'Historical WUENIC values reflect revisions, not necessarily the data available then.\n'
        'Historic incidence is embedded in surge definition; label and predictor definitions are mathematically linked.\n'
        'No causal claims and no claims of WHO-confirmed outbreak prediction.\n',encoding='utf-8')
    print('\nSTEP 10: ADVANCED ML MODEL VALIDATION (NO TEST EVALUATION)\n')
    print(pd.DataFrame(audits).to_string(index=False))
    print('\nAdvanced model validation metrics:\n')
    print(metrics_df[['cohort','model','n','events','AUROC','average_precision',
                      'brier_score','recall_at_10pct']].round(4).to_string(index=False))
    print(f'\nReports saved: {OUT.relative_to(ROOT)}/')
    print('NOTE: No test data used; provisional outcome; validation comparisons exploratory.')

if __name__=='__main__':
    main()
