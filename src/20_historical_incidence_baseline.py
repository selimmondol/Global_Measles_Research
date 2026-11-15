#!/usr/bin/env python3
"""STEP 20 — EXPLORATORY POST-TEST historical-incidence-only baselines.

Run from your existing Measles project root:
    python src/20_historical_incidence_baseline.py

Uses existing Step-15 independent-test predictions and frozen Step-6 feature
values, or Step-3 candidates when the Step-6 file is unavailable. NO models
are retrained, no test labels altered, no frozen files modified, and no
post-test selection of a 'best' baseline is permitted.

Two separate, predeclared descriptive ranking rules:
    (A) prior-year incidence per million (primary simple baseline)
    (B) median incidence in prior three years (sensitivity baseline)

Important: these heuristics are not necessarily optimal early-warning scores;
the surge outcome itself uses prior incidence median as an ingredient. Results
are post-test exploratory, NOT newly independent confirmation.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = 'surge_C_2x_5pm'
SEED = 20261110
N_BOOT = 200
CAPACITIES = (10, 20)
PREDICTION_SCORES = ('coverage_shortfall_rank', 'rf_regularized', 'xgb_regularized')
BASELINES = ('incidence_lag1', 'incidence_median_3y')
EXPECTED = {'test': 505, 'unseen_country_test': 119}


def locate_root(cli_root):
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    script = Path(__file__).resolve()
    return script.parent.parent if script.parent.name == 'src' else script.parent


def read_existing_predictions(root):
    p = root / 'results/independent_test_v1/test_predictions.csv'
    if not p.is_file():
        raise FileNotFoundError(f'Missing Step-15 predictions: {p}\nRun Step 15 using the ORIGINAL frozen protocol; do not overwrite an existing independent test.')
    d = pd.read_csv(p, low_memory=False)
    need = {'partition','Country','Year','y',*PREDICTION_SCORES}
    if need - set(d.columns):
        raise ValueError(f'Step-15 predictions missing: {sorted(need-set(d.columns))}')
    if d.duplicated(['partition','Country','Year']).any():
        raise ValueError('Duplicate country-year in Step-15 predictions')
    if d.partition.value_counts().to_dict() != EXPECTED:
        raise ValueError(f'Unexpected test partition sizes: {d.partition.value_counts().to_dict()}')
    if not d.Year.between(2022,2025).all() or not d.y.isin([0,1]).all():
        raise ValueError('Unexpected years/outcomes in Step-15 predictions')
    if any(d.loc[d.partition.eq(k),'y'].sum() != n for k,n in [('test',143),('unseen_country_test',33)]):
        raise ValueError('Outcome positive counts differ from verified Step-15 outputs')
    return d


def read_historical_scores(root):
    feat = root / 'data/processed/Measles_Model_Features.csv'
    candidates = root / 'data/processed/Measles_Resurgence_Candidates.csv'
    needed = {'Country','Year','incidence_lag1','incidence_median_3y'}
    if feat.is_file():
        f = pd.read_csv(feat,low_memory=False)
        if needed.issubset(f.columns) and len(f) == 5642:
            return f, str(feat.relative_to(root))
    # Fully deterministic fallback: Step-3 audit already stores the previous-year
    # incidence and prior-3y median computed without target-year values.
    if not candidates.is_file():
        raise FileNotFoundError(f'Neither full {feat} nor {candidates} exists. Recreate Step 6 from frozen inputs if necessary.')
    f = pd.read_csv(candidates,low_memory=False)
    mapping = {'prior_3y_median_per_million':'incidence_median_3y'}
    if not {'Country','Year','incidence_lag1','prior_3y_median_per_million',TARGET}.issubset(f):
        raise ValueError('Step-3 candidate columns missing or unexpectedly renamed')
    f = f.rename(columns=mapping)
    if len(f) != 5642:
        raise ValueError('Unexpected Step-3 row count')
    return f, str(candidates.relative_to(root))


def attach_scores(pred, history):
    cols = ['Country','Year',*BASELINES]
    if history.duplicated(['Country','Year']).any():
        raise ValueError('Historical features have duplicate country-years')
    extra = [TARGET] if TARGET in history.columns else []
    merged = pred.merge(history[cols + extra],on=['Country','Year'],how='left',validate='one_to_one',indicator=True)
    if not merged._merge.eq('both').all():
        raise ValueError('Historical feature rows missing for tested country-years')
    merged = merged.drop(columns='_merge')
    if TARGET in merged.columns:
        if not merged[TARGET].eq(merged.y).all():
            raise ValueError('Step-3 outcome differs from frozen Step-15 outcome. STOP.')
        merged = merged.drop(columns=TARGET)
    for col in BASELINES:
        merged[col] = pd.to_numeric(merged[col],errors='raise')
        if merged[col].isna().any() or not np.isfinite(merged[col]).all():
            raise ValueError(f'Missing/invalid historical score {col} among outcome-eligible test observations')
    for col in PREDICTION_SCORES:
        if merged[col].isna().any():
            raise ValueError(f'Missing existing score: {col}')
    return merged


def rank_partition(d, score, cap):
    """Annual top-K, numpy sorting with alphabetical country and replicate ties."""
    events = int(d['y'].sum())
    selected = detected = 0
    flags = []
    yrs=d['Year'].to_numpy()
    vals=d[score].to_numpy()
    country=d['Country'].to_numpy(dtype=str)
    rep=d['_replicate'].to_numpy()
    labels=d['y'].to_numpy(dtype=int)
    for year in np.unique(yrs):
        ids=np.flatnonzero(yrs == year)
        k=math.ceil(len(ids)*cap/100)
        # lexsort final key is primary; negative risk sorts descending.
        order=np.lexsort((rep[ids],country[ids],-vals[ids]))
        found=int(labels[ids[order[:k]]].sum())
        e=int(labels[ids].sum())
        selected+=k;detected+=found
        flags.append((int(year),len(ids),k,e,found))
    return {
        'evaluated':len(d),'selected':selected,'events':events,
        'detected':detected,'missed':events-detected,
        'false_alarms':selected-detected,
        'recall':detected/events if events else np.nan,
        'precision':detected/selected if selected else np.nan,
    },flags


def resample_countries(d, rng):
    # Preserve every year for each sampled country; duplicate countries are
    # distinct sampled clusters using _replicate as a deterministic tie key.
    by_country = {country:np.flatnonzero(d['Country'].to_numpy() == country)
                  for country in sorted(d['Country'].unique())}
    names = np.asarray(sorted(by_country))
    picked=rng.choice(names,size=len(names),replace=True)
    ids=np.concatenate([by_country[c] for c in picked])
    reps=np.concatenate([np.full(len(by_country[c]),i,dtype=int)
                         for i,c in enumerate(picked)])
    result=d.iloc[ids].copy()
    result['_replicate']=reps
    return result


def percentile(x):
    arr=np.asarray(x,dtype=float)
    arr=arr[np.isfinite(arr)]
    return (float(np.quantile(arr,0.025)),float(np.quantile(arr,0.975))) if len(arr)>20 else (np.nan,np.nan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',default=None,help='Measles project root (normally unnecessary when placed in src/)')
    args=parser.parse_args()
    root=locate_root(args.project_root)
    pred=read_existing_predictions(root)
    history,source=read_historical_scores(root)
    data=attach_scores(pred, history)
    data['_replicate']=0
    summary=[]; annual=[]; paired=[]
    scores=(*PREDICTION_SCORES,*BASELINES)
    for partition, sub in data.groupby('partition',sort=True):
        point={}
        for cap in CAPACITIES:
            for score in scores:
                res,byyear=rank_partition(sub,score,cap)
                summary.append({'partition':partition,'approach':score,'capacity_pct':cap,**res})
                point[(cap,score)]=res['recall']
                for year,n,k,e,found in byyear:
                    annual.append({'partition':partition,'approach':score,'capacity_pct':cap,
                                   'Year':year,'evaluated':n,'selected':k,'events':e,
                                   'detected':found,'recall':found/e if e else np.nan})
        # Confirm the exact prior Step-15 decisions, preventing mismatched ranking.
        original=root/'results/independent_test_v1/pooled_decision_summary.csv'
        if original.exists():
            frozen=pd.read_csv(original)
            for cap in CAPACITIES:
                for score in PREDICTION_SCORES:
                    match=frozen.loc[(frozen.partition==partition)&(frozen.approach==score)&(frozen.capacity_pct==cap)]
                    got=next(row for row in summary if row['partition']==partition and row['capacity_pct']==cap and row['approach']==score)
                    if len(match)!=1 or int(match.iloc[0].detected)!=got['detected'] or int(match.iloc[0].selected)!=got['selected']:
                        raise ValueError('Mismatch with frozen Step 15 selection: '+str((partition,score,cap)))
        rng=np.random.default_rng(SEED + (0 if partition=='test' else 1))
        comparisons=[(baseline,other) for baseline in BASELINES for other in ('coverage_shortfall_rank','rf_regularized')]
        comparisons.extend([('rf_regularized',baseline) for baseline in BASELINES])
        for cap in CAPACITIES:
            diffs={pair:[] for pair in comparisons}
            for _ in range(N_BOOT):
                b=resample_countries(sub,rng)
                values={s:rank_partition(b,s,cap)[0]['recall'] for s in scores}
                for a,z in comparisons:
                    diffs[(a,z)].append(values[a]-values[z])
            for a,z in comparisons:
                lo,hi=percentile(diffs[(a,z)])
                paired.append({'partition':partition,'capacity_pct':cap,'model_a':a,'model_b':z,
                               'observed_difference':point[(cap,a)]-point[(cap,z)],
                               'lower_95pct':lo,'upper_95pct':hi,'bootstrap_draws':N_BOOT})
    out=root/'results/posttest_historical_incidence_baseline_v1'
    if out.exists():
        raise FileExistsError(f'Output directory already exists; refusing to overwrite: {out}')
    out.mkdir(parents=True)
    pd.DataFrame(summary).to_csv(out/'historical_baseline_pooled.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'historical_baseline_annual.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'historical_baseline_paired_bootstrap.csv',index=False)
    (out/'analysis_notes.txt').write_text(
        f'EXPLORATORY POST-TEST ANALYSIS, not new independent validation.\n'
        f'Historical score source: {source}\n'
        'Two separately reported simple baselines: prior-year incidence and prior-three-year median.\n'
        'No fitting, tuning, threshold search, refitting, or protocol modification.\n'
        'Target-year incidence never enters ranking scores.\n'
        'Outcome itself uses historical incidence median: potential mechanical advantage to history-based ranking.\n'
        '200 paired country-cluster bootstrap replicates per capacity/partition, fixed seed.\n'
        'Confidence intervals do not account for shared global calendar-year shocks.\n'
        'Only retrospectively outcome-evaluable country-years were ranked.\n'
        'Do not select whichever historical rule looks best on the held-out test.\n',encoding='utf-8')
    print('\nSTEP 20: POST-TEST HISTORICAL INCIDENCE BASELINE (EXPLORATORY)')
    view=pd.DataFrame(summary)
    print(view.loc[(view.capacity_pct==10),['partition','approach','events','selected','detected','recall','precision']].round(4).to_string(index=False))
    print('\nPaired 95% country-bootstrap CIs (RF minus historical baselines at 10%):')
    p=pd.DataFrame(paired)
    print(p.loc[(p.capacity_pct==10)&(p.model_a=='rf_regularized'),
                ['partition','model_b','observed_difference','lower_95pct','upper_95pct']].round(4).to_string(index=False))
    print(f'\nSaved files in {out}')

if __name__ == '__main__':
    main()
