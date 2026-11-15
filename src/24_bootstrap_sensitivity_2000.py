#!/usr/bin/env python3
"""Step 16: post-test reliability audit of frozen Measles predictions.

Run from project root: python src/16_test_reliability.py
Reads ONLY existing Step-15 predictions/reports. Does not refit, tune, edit labels,
or modify the frozen protocol. Bootstrap intervals are exploratory post-test.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'results' / 'independent_test_v1'
OUTPUT = ROOT / 'results' / 'bootstrap_sensitivity_2000'
PRED = INPUT / 'test_predictions.csv'
ANNUAL = INPUT / 'annual_decision_metrics.csv'
POOLED = INPUT / 'pooled_decision_summary.csv'
RECEIPT = INPUT / 'evaluation_receipt.json'
MODEL_SCORES = {
    'coverage_shortfall_rank':'coverage_shortfall_rank',
    'vaccination_only':'vaccination_only',
    'combined':'combined',
    'rf_regularized':'rf_regularized',
    'xgb_regularized':'xgb_regularized',
}
CAPACITIES=(10,20)
BOOTSTRAPS=2000
SEED=20261110


def evaluate_rank(df, score_col, capacity):
    total_events=int(df['y'].sum())
    selected=0; detected=0
    for _, g in df.groupby('Year', sort=True):
        k=math.ceil(len(g)*capacity/100)
        # Stable alphabetical tie rule, matching frozen protocol.
        g=g.sort_values([score_col,'Country','_replicate'], ascending=[False, True, True], kind='mergesort')
        selected+=k
        detected+=int(g.head(k)['y'].sum())
    return dict(evaluated=len(df),selected=selected,events=total_events,
                detected=detected,missed=total_events-detected,
                false_alarms=selected-detected,
                recall=(detected/total_events if total_events else np.nan),
                precision=(detected/selected if selected else np.nan))


def bootstrap_partition(d, rng):
    names=np.array(sorted(d['Country'].unique()))
    picked=rng.choice(names,size=len(names),replace=True)
    pieces=[]
    groups={name:g for name,g in d.groupby('Country',sort=False)}
    for i, name in enumerate(picked):
        v=groups[name].copy()
        # each resampled country copy is a distinct bootstrap unit
        v['_replicate']=i
        pieces.append(v)
    return pd.concat(pieces,ignore_index=True)


def percentile_interval(vals):
    a=np.asarray(vals,dtype=float)
    a=a[np.isfinite(a)]
    return (float(np.quantile(a,0.025)), float(np.quantile(a,0.975))) if len(a)>20 else (np.nan,np.nan)


def main():
    for f in (PRED,ANNUAL,POOLED,RECEIPT):
        if not f.exists():
            raise FileNotFoundError(f'Missing Step 15 report: {f}')
    with RECEIPT.open(encoding='utf-8') as f:
        receipt=json.load(f)
    if receipt.get('frozen_manifest_files_checked') != 28 or 'once' not in receipt.get('test_access',''):
        raise ValueError('Unexpected Step 15 receipt; verify frozen protocol manually.')
    d=pd.read_csv(PRED)
    required={'Country','Year','partition','y','mcv1_lag1','mcv2_lag1',*MODEL_SCORES.values()}
    missing=required-set(d.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    if d.duplicated(['partition','Country','Year']).any(): raise ValueError('Duplicate country-year predictions.')
    if set(d['partition'].unique()) != {'test','unseen_country_test'}:
        raise ValueError('Unexpected partitions in predictions.')
    if not d['Year'].between(2022,2025).all(): raise ValueError('Wrong evaluation years.')
    if not d['y'].isin([0,1]).all(): raise ValueError('Unexpected outcome labels.')
    if d[list(MODEL_SCORES.values())].isna().any().any():
        raise ValueError('Missing risk scores; do not silently impute.')
    d['_replicate']=0
    OUTPUT.mkdir(parents=True,exist_ok=True)
    original_annual=pd.read_csv(ANNUAL)
    original_pooled=pd.read_csv(POOLED)
    annual=[]; pooled=[]; classification=[]; high=[]; boot=[]; paired=[]
    rng=np.random.default_rng(SEED)
    for partition, p in d.groupby('partition',sort=True):
        for approach, col in MODEL_SCORES.items():
            for capacity in CAPACITIES:
                summary=evaluate_rank(p,col,capacity)
                pooled.append(dict(partition=partition,approach=approach,capacity_pct=capacity,**summary))
                for year,g in p.groupby('Year',sort=True):
                    annual.append(dict(partition=partition,Year=year,approach=approach,capacity_pct=capacity,
                                       **evaluate_rank(g,col,capacity)))
        for model in ('vaccination_only','combined','rf_regularized','xgb_regularized'):
            classification.append(dict(partition=partition,model=model,n=len(p),events=int(p.y.sum()),
                AUROC=roc_auc_score(p.y,p[model]),
                average_precision=average_precision_score(p.y,p[model]),
                brier_score=brier_score_loss(p.y,p[model])))
        subset=p[(p.mcv1_lag1>=95)&(p.mcv2_lag1>=95)]
        for approach,col in MODEL_SCORES.items():
            # Important: select top K among ALL countries first; then measure coverage of high-vax events.
            tmp=p.copy(); tmp['_selected']=False
            for y,g in p.groupby('Year'):
                k=math.ceil(len(g)*0.1)
                ix=g.sort_values([col,'Country'],ascending=[False,True],kind='mergesort').head(k).index
                tmp.loc[ix,'_selected']=True
            h=tmp[(tmp.mcv1_lag1>=95)&(tmp.mcv2_lag1>=95)]
            n_events=int(h.y.sum()); detected=int(h.loc[h._selected,'y'].sum())
            high.append(dict(partition=partition,approach=approach,high_vax_rows=len(h),
                high_vax_events=n_events,high_vax_detected=detected,
                high_vax_missed=n_events-detected,recall_in_high_vax=(detected/n_events if n_events else np.nan)))
        comparisons=[('rf_regularized','coverage_shortfall_rank'),
                     ('rf_regularized','vaccination_only'),
                     ('rf_regularized','combined'),
                     ('rf_regularized','xgb_regularized'),
                     ('xgb_regularized','coverage_shortfall_rank')]
        for capacity in CAPACITIES:
            samples={m:[] for m in MODEL_SCORES}
            differences={pair:[] for pair in comparisons}
            for _ in range(BOOTSTRAPS):
                b=bootstrap_partition(p,rng)
                values={m:evaluate_rank(b,col,capacity)['recall'] for m,col in MODEL_SCORES.items()}
                for m,v in values.items(): samples[m].append(v)
                for (a,z) in comparisons: differences[(a,z)].append(values[a]-values[z])
            for m, col in MODEL_SCORES.items():
                lo,hi=percentile_interval(samples[m])
                boot.append(dict(partition=partition,approach=m,capacity_pct=capacity,
                    observed_recall=evaluate_rank(p,col,capacity)['recall'],lower_95pct=lo,upper_95pct=hi,
                    bootstrap_draws=BOOTSTRAPS))
            for a,z in comparisons:
                lo,hi=percentile_interval(differences[(a,z)])
                obs=evaluate_rank(p,MODEL_SCORES[a],capacity)['recall']-evaluate_rank(p,MODEL_SCORES[z],capacity)['recall']
                paired.append(dict(partition=partition,capacity_pct=capacity,model_a=a,model_b=z,
                   observed_difference=obs,lower_95pct=lo,upper_95pct=hi,
                   bootstrap_draws=BOOTSTRAPS))
    yearly=pd.DataFrame(annual); pool=pd.DataFrame(pooled); classif=pd.DataFrame(classification)
    orig=original_pooled.copy()
    checks=[]
    for _,row in pool.iterrows():
        q=orig[(orig.partition==row.partition)&(orig.approach==row.approach)&(orig.capacity_pct==row.capacity_pct)]
        if len(q)!=1: raise ValueError(f'Step 15 pooled row missing or duplicated: {row.partition}, {row.approach}, {row.capacity_pct}')
        for fld in ('evaluated','selected','events','detected','missed','false_alarms'):
            if int(row[fld])!=int(q.iloc[0][fld]):
                raise ValueError(f'Step 15 pooled mismatch: {row.partition}, {row.approach}, {fld}')
        checks.append({'partition':row.partition,'approach':row.approach,'capacity_pct':row.capacity_pct,'status':'MATCH'})
    for _,row in yearly.iterrows():
        q=original_annual[(original_annual.partition==row.partition)&(original_annual.approach==row.approach)&
                          (original_annual.capacity_pct==row.capacity_pct)&(original_annual.Year==row.Year)]
        if len(q)!=1 or int(q.iloc[0].detected)!=int(row.detected):
            raise ValueError(f'Step 15 annual mismatch: {row.partition}, {row.approach}, {row.Year}')
    outputs={'annual_metrics_verified.csv':yearly,'pooled_metrics_verified.csv':pool,
        'classification_metrics_recomputed.csv':classif,'high_vaccination_detection.csv':pd.DataFrame(high),
        'bootstrap_recall_intervals.csv':pd.DataFrame(boot),
        'paired_bootstrap_differences.csv':pd.DataFrame(paired),
        'step15_alignment_checks.csv':pd.DataFrame(checks)}
    for name,frame in outputs.items(): frame.to_csv(OUTPUT/name,index=False)
    notes='''STEP 16: POST-TEST DESCRIPTIVE RELIABILITY AUDIT\n\nNo models were fitted, tuned or recalibrated. No frozen files modified.\nThe independent tests have ALREADY been opened at Step 15; these are post-test analyses, not a new confirmation.\nBootstrap: sample countries with replacement WITHIN each partition; retain their annual observations; recompute per-year top-K.\nSmall number of calendar years (4) and shared global shocks are NOT accounted for by country-only bootstrap.\nCountry bootstrap duplicates are treated as sampled copies; inference is conditional on the evaluated-country population.\nRisk ranking is restricted to retrospectively outcome-evaluable country-years.\nHigher dose coverage uses >=95% prior-year national WUENIC estimates, not measured immunity.\nHigh-vaccination detection is computed AFTER overall annual prioritization (no subgroup-specific K).\nDefinition C is researcher-defined incidence surge, not WHO-confirmed outbreak.\nOther endpoint robustness (A/B/onset/zero-baseline excluded) requires joining frozen predictions to\noriginal labels WITHOUT model changes; it is not estimated by this script.\nNo claims of prospective performance; WUENIC current revision is retrospective.\n'''
    (OUTPUT/'reliability_notes.txt').write_text(notes,encoding='utf-8')
    print('STEP 16: POST-TEST RELIABILITY (NO TRAINING / NO TUNING)')
    print(pool[(pool.capacity_pct==10)&(pool.approach.isin(['coverage_shortfall_rank','rf_regularized','xgb_regularized']))]
       [['partition','approach','selected','events','detected','missed','recall']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\nCountry-cluster bootstrap comparisons, 10% capacity:')
    print(pd.DataFrame(paired).query('capacity_pct == 10 and model_b == "coverage_shortfall_rank"')
          [['partition','model_a','observed_difference','lower_95pct','upper_95pct']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print(f'\nSaved: {OUTPUT.relative_to(ROOT)}')
    print('NOTE: POST-TEST descriptive analysis; no claim of new independent validation.')

if __name__=='__main__': main()
