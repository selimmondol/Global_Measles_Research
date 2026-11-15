#!/usr/bin/env python3
"""STEP 15 — ONE-TIME independent evaluation, AFTER Step 14 protocol freeze.

Run from Measles project root:
    python src/15_independent_test_evaluation.py

Guards: verify frozen SHA256 source/code checksums, exact protocol, split sizes,
previously unseen-country separation, and write ONLY to a new output directory.
Models are fit ONCE to 2003–2018 training rows (no validation refitting); the
2022–2025 test outcomes are opened only after protocol-integrity checks.
This is a retrospective analysis of researcher-defined annual incidence surges,
NOT WHO-confirmed outbreaks nor real-time early-warning performance.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / 'results/protocol_freeze_v1/frozen_protocol.json'
OUT = ROOT / 'results/independent_test_v1'
FEATURE = ROOT / 'data/processed/Measles_Model_Features.csv'
SPLIT = ROOT / 'results/validation_design/split_assignments.csv'
TARGET = 'surge_C_2x_5pm'
FEATURES = ['mcv1_lag1', 'mcv2_lag1', 'incidence_lag1', 'incidence_median_3y', 'incidence_std_3y']
SEED = 42
CAPACITIES = [0.10, 0.20]
MODELS = ['coverage_shortfall_rank','vaccination_only','combined','rf_regularized','xgb_regularized']
EXPECTED = {'train':2144, 'validation':396, 'test':505,
            'unseen_country_test':119, 'reserved_not_used':600}


def stop(msg):
    raise SystemExit('STOP — no independent test generated: ' + str(msg))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def integrity_check():
    if OUT.exists():
        stop(f'{OUT} already exists. Never overwrite test results or rerun for model selection.')
    if not LOCK.is_file():
        stop('Frozen protocol missing. Run Step 14 before independent testing.')
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    protocol = lock.get('protocol', {})
    if lock.get('protocol', {}).get('status') == 'FROZEN_BEFORE_INDEPENDENT_TEST':
        # tolerate status at top or within protocol? Step 14 uses top-level protocol.status.
        pass
    if protocol.get('status') != 'FROZEN_BEFORE_INDEPENDENT_TEST':
        stop('Protocol is not in frozen status.')
    checks = [
        (protocol.get('cohort'), 'all_eligible_imputed_vaccination'),
        (protocol.get('target', {}).get('column'), TARGET),
        (protocol.get('models', {}).get('primary'), 'rf_regularized'),
        (protocol.get('models', {}).get('secondary'), 'xgb_regularized'),
        (protocol.get('models', {}).get('statistical_baseline'), 'combined'),
        (protocol.get('models', {}).get('policy_baseline'), 'coverage_shortfall_rank'),
        (protocol.get('primary_metric'), 'Pooled annual event recall at 10% selection capacity'),
        (protocol.get('train_years'), [2003,2018]),
        (protocol.get('validation_years'), [2019,2021]),
        (protocol.get('locked_future_years'), [2022,2025]),
        (protocol.get('held_out_country_count'), 40),
        (protocol.get('no_test_tuning'), True),
        (protocol.get('features'), FEATURES),
    ]
    for actual, required in checks:
        if actual != required:
            stop(f'Protocol field inconsistent: found {actual!r}, expected {required!r}')
    frozen_files = lock.get('input_and_code_sha256', {})
    if len(frozen_files) < 20:
        stop('Frozen file manifest unexpectedly short.')
    for rel, detail in frozen_files.items():
        p = ROOT / rel
        if not p.is_file():
            stop(f'Missing frozen file: {rel}')
        if sha256(p) != detail['sha256']:
            stop(f'Frozen input/script CHANGED since Step 14: {rel}. Restore frozen copy; do NOT overwrite protocol.')
    return lock


def read_cohorts():
    # Split metadata first; ensure no test labels are consulted until integrity verified.
    splits = pd.read_csv(SPLIT, low_memory=False)
    required = {'Country','Year','evaluation_role','eligible_for_outcome'}
    if not required.issubset(splits):
        stop('Unexpected split schema')
    if len(splits) != 5642 or splits.duplicated(['Country','Year']).any():
        stop('Split keys/size incorrect')
    splits['eligible_for_outcome'] = splits['eligible_for_outcome'].astype('boolean')
    counts = splits.loc[splits.eligible_for_outcome.fillna(False),'evaluation_role'].value_counts().to_dict()
    for role,n in EXPECTED.items():
        if counts.get(role) != n:
            stop(f'Eligibility count for {role} changed: {counts.get(role)} != {n}')
    for role,lo,hi in [('train',2003,2018),('validation',2019,2021),
                       ('test',2022,2025),('unseen_country_test',2022,2025)]:
        if not splits.loc[splits.evaluation_role.eq(role),'Year'].between(lo,hi).all():
            stop(f'Unexpected year partition: {role}')
    dev_countries=set(splits.loc[splits.evaluation_role.isin(['train','validation','test']),'Country'])
    reserved=set(splits.loc[splits.evaluation_role.eq('unseen_country_test'),'Country'])
    if len(reserved) != 40 or dev_countries & reserved:
        stop('Unseen-country holdout violated')
    if splits.evaluation_role.eq('unseen_country_test').sum() == 0:
        stop('No unseen-country test')

    features=pd.read_csv(FEATURE,low_memory=False)
    required_feat={'Country','Year','outcome_eligible',TARGET,*FEATURES}
    if not required_feat.issubset(features):
        stop(f'Missing feature columns: {sorted(required_feat-set(features))}')
    if features.duplicated(['Country','Year']).any():
        stop('Duplicate features')
    data=splits[['Country','Year','evaluation_role','eligible_for_outcome']].merge(
        features[['Country','Year','outcome_eligible',TARGET,*FEATURES]],
        on=['Country','Year'],validate='one_to_one',how='left',indicator=True)
    if not data['_merge'].eq('both').all():
        stop('Some split countries/years absent from feature data')
    if not data.eligible_for_outcome.equals(data.outcome_eligible.astype('boolean')):
        stop('Eligibility mismatch between frozen split and features')
    data=data.loc[data.eligible_for_outcome.fillna(False)].copy()
    if not data[TARGET].isin([0,1]).all():
        stop('Invalid eligible outcome labels')
    for feature in FEATURES:
        data[feature]=pd.to_numeric(data[feature],errors='raise').replace([np.inf,-np.inf],np.nan)
    if not data.loc[data.evaluation_role.eq('train'),'Year'].between(2003,2018).all():
        stop('Training period corrupted')
    return data


def make_models():
    return {
        'vaccination_only': (['mcv1_lag1','mcv2_lag1'], Pipeline([
            ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('scaler',StandardScaler()),
            ('logit',LogisticRegression(max_iter=2000,solver='lbfgs',random_state=SEED))])),
        'combined': (FEATURES, Pipeline([
            ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('scaler',StandardScaler()),
            ('logit',LogisticRegression(max_iter=2000,solver='lbfgs',random_state=SEED))])),
        'rf_regularized': (FEATURES, Pipeline([
            ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('classifier',RandomForestClassifier(n_estimators=250,min_samples_leaf=10,
                              max_features='sqrt',max_depth=5,random_state=SEED,n_jobs=1))])),
        'xgb_regularized': (FEATURES, Pipeline([
            ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('classifier',XGBClassifier(n_estimators=180,max_depth=2,learning_rate=0.04,
                  subsample=0.85,colsample_bytree=0.85,min_child_weight=10,
                  reg_lambda=10,objective='binary:logistic',eval_metric='logloss',
                  random_state=SEED,tree_method='hist',n_jobs=1))])),
    }


def annual_selection(df, score_col, cap):
    """Rank country-years inside each year, with fixed Country ascending tie break."""
    results=[]; rows=[]
    for year, sub in df.groupby('Year',sort=True):
        if sub.Country.duplicated().any():
            stop(f'Duplicate country in year {year}')
        n=len(sub); k=math.ceil(cap*n)
        ordered=sub.sort_values([score_col,'Country'],ascending=[False,True],kind='stable').copy()
        ordered['rank_within_year']=np.arange(1,n+1)
        ordered['selected']=ordered.rank_within_year.le(k)
        event=int(ordered.y.sum()); caught=int(ordered.loc[ordered.selected,'y'].sum())
        results.append({'Year':int(year),'capacity_pct':int(cap*100), 'evaluated':n,'selected':k,
                        'events':event,'detected':caught,'missed':event-caught,
                        'false_alarms':k-caught,'recall':caught/event if event else np.nan,
                        'precision':caught/k})
        rows.append(ordered.loc[ordered.selected|ordered.y.eq(1),
                                ['Country','Year','y','rank_within_year','selected',score_col]].rename(
                                    columns={score_col:'ranking_score'}))
    return results,pd.concat(rows,ignore_index=True)


def statistical_metrics(y,p):
    if np.unique(y).size != 2:
        return {'AUROC':np.nan,'average_precision':np.nan,'brier_score':float(brier_score_loss(y,p))}
    return {'AUROC':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'brier_score':float(brier_score_loss(y,p))}


def main():
    lock=integrity_check()
    # NOW opens test labels, for the first and only independent test run.
    data=read_cohorts()
    train=data.loc[data.evaluation_role.eq('train')].sort_values(['Country','Year']).copy()
    if len(train)!=2144 or train[TARGET].nunique()!=2:
        stop('Unexpected training set')
    tests={role:data.loc[data.evaluation_role.eq(role)].sort_values(['Country','Year']).copy()
           for role in ['test','unseen_country_test']}
    for role,s in tests.items():
        if len(s)!=EXPECTED[role] or not s[TARGET].isin([0,1]).all():
            stop(f'{role} changed or is empty')
    models=make_models()
    # Configs mirror frozen Steps 8 and 10, fitted on TRAIN only.
    for name,(cols,pipeline) in models.items():
        pipeline.fit(train[cols],train[TARGET].astype(int))

    score_rows=[]; yearly_rows=[]; decision_rows=[]; predictions=[]
    for role,test in tests.items():
        base=test[['Country','Year',TARGET,'mcv1_lag1','mcv2_lag1']].copy()
        base=base.rename(columns={TARGET:'y'})
        base['y']=base.y.astype(int)
        vax_known=base[['mcv1_lag1','mcv2_lag1']].notna().all(axis=1)
        shortfall=100-base[['mcv1_lag1','mcv2_lag1']].min(axis=1)
        base['coverage_shortfall_rank']=shortfall.where(vax_known,-np.inf)
        # Missing coverage ranked last, not imputed to low-vaccination.
        for name,(cols,pipeline) in models.items():
            base[name]=pipeline.predict_proba(test[cols])[:,1]
        for row in base.itertuples(index=False):
            predictions.append({'partition':role,**row._asdict()})
        for model in MODELS:
            if model!='coverage_shortfall_rank':
                stats=statistical_metrics(base.y.to_numpy(),base[model].to_numpy())
                score_rows.append({'partition':role,'model':model,'n':len(base),
                                   'events':int(base.y.sum()),**stats})
            for cap in CAPACITIES:
                year_results,selected=annual_selection(base,model,cap)
                for record in year_results:
                    record.update({'partition':role,'approach':model})
                    yearly_rows.append(record)
                selected['partition']=role
                selected['approach']=model
                selected['capacity_pct']=int(cap*100)
                decision_rows.append(selected)
    year_df=pd.DataFrame(yearly_rows)
    by=['partition','approach','capacity_pct']
    pooled=year_df.groupby(by,as_index=False)[['evaluated','selected','events','detected','missed','false_alarms']].sum()
    pooled['recall']=pooled.detected / pooled.events.replace(0,np.nan)
    pooled['precision']=pooled.detected / pooled.selected.replace(0,np.nan)
    pooled['false_alarm_share']=pooled.false_alarms/pooled.selected.replace(0,np.nan)
    all_predictions=pd.DataFrame(predictions)
    if any((~all_predictions.loc[all_predictions.partition.eq(role),'Year'].between(2022,2025)).any()
           for role in tests):
        stop('Unexpected test years in predictions')

    # No overwrite even if process was interrupted after creating the directory.
    OUT.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(score_rows).to_csv(OUT/'test_classification_metrics.csv',index=False)
    year_df.to_csv(OUT/'annual_decision_metrics.csv',index=False)
    pooled.to_csv(OUT/'pooled_decision_summary.csv',index=False)
    all_predictions.to_csv(OUT/'test_predictions.csv',index=False)
    pd.concat(decision_rows,ignore_index=True).to_csv(OUT/'selected_and_missed_test_rows.csv',index=False)
    receipt={'executed_utc':datetime.now(timezone.utc).isoformat(),
             'frozen_protocol_sha256':sha256(LOCK),
             'frozen_manifest_files_checked':len(lock['input_and_code_sha256']),
             'model_refit_policy':'2003-2018 TRAIN ONLY; no validation rows in fit',
             'test_access':'2022-2025 partition labels evaluated exactly once in this run',
             'eligibility_note':'Ranking limited to retrospectively outcome-evaluable country-years',
             'not_realtime':'WUENIC historical revisions not known to be available at prediction time'}
    (OUT/'evaluation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (OUT/'limitations.txt').write_text(
        'Annual reported-incidence surge (researcher-defined Definition C), NOT WHO-confirmed measles outbreaks.\n'
        'Test rankings use retrospectively eligible outcome rows; deployment selection may differ.\n'
        'Current-vintage WUENIC series are retrospectively revised. No prospective prediction claim.\n'
        'The holdout is country-held-out plus future-year; temporal/global shocks may affect both tests.\n'
        'Only 2003–2018 training fitted. No tuning or post-test model modification permitted.\n'
        'The 40 unseen countries were reserved entirely, including historical observations.\n'
        'Inference, CI, onset and missingness sensitivities require separately labelled analyses.\n',encoding='utf-8')
    print('\nSTEP 15: INDEPENDENT TEST EVALUATION (FROZEN PROTOCOL)')
    print('Freeze manifest hashes verified:',len(lock['input_and_code_sha256']))
    print('TRAIN only:',len(train),'rows | test:',len(tests['test']),
          '| unseen-country test:',len(tests['unseen_country_test']))
    print('\nPrimary decision comparison, annual Recall@10%:')
    print(pooled.loc[pooled.capacity_pct.eq(10),
                     ['partition','approach','evaluated','selected','events','detected','missed','false_alarms','recall']].round(4).to_string(index=False))
    print('\nHeld-out classification metrics:')
    print(pd.DataFrame(score_rows).round(4).to_string(index=False))
    print('\nOutputs:',OUT.relative_to(ROOT))
    print('ONE-TIME test completed; do not tune or replace frozen protocol on these outcomes.')

if __name__=='__main__':
    main()
