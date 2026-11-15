#!/usr/bin/env python3
"""Step 17: POST-TEST outcome robustness of frozen measles predictions.

Run from the Measles project root:
    python src/17_outcome_robustness.py

Reads ONLY the Step 15 saved predictions + Step 3 candidate target file.
No training, tuning, imputation, feature edits or changes to frozen protocol.
These are post-test exploratory checks, NOT additional independent validation.
"""
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / 'results/independent_test_v1/test_predictions.csv'
RECEIPT = ROOT / 'results/independent_test_v1/evaluation_receipt.json'
TARGETS = ROOT / 'data/processed/Measles_Resurgence_Candidates.csv'
OUT = ROOT / 'results/outcome_robustness_posttest_v1'
SCORES = ['coverage_shortfall_rank', 'vaccination_only', 'combined', 'rf_regularized', 'xgb_regularized']
MODEL_SCORES = ['vaccination_only', 'combined', 'rf_regularized', 'xgb_regularized']
MAIN = 'surge_C_2x_5pm'


def number_bool(x):
    """Interpret the explicit True/False eligibility column safely."""
    z = x.astype('string').str.strip().str.lower()
    bad = z.notna() & ~z.isin(['true','false','1','0'])
    if bad.any():
        raise ValueError(f'Unexpected eligibility values: {z[bad].unique().tolist()}')
    return z.isin(['true','1'])


def evaluate_annual(d, col, pct):
    """Re-rank frozen risk scores annually within each outcome-evaluable subset."""
    selected = 0
    detected = 0
    for year, g in d.groupby('Year', sort=True):
        k = math.ceil(len(g)*pct/100)
        ranked = g.sort_values([col,'Country'], ascending=[False,True], kind='mergesort')
        selected += k
        detected += int(ranked.head(k)['target'].sum())
    events = int(d['target'].sum())
    return {'evaluated':len(d), 'selected':selected, 'events':events,
            'detected':detected, 'missed':events-detected,
            'false_alarms':selected-detected,
            'recall':detected/events if events else np.nan,
            'precision':detected/selected if selected else np.nan}


def classification(d, col):
    y = d['target'].to_numpy(dtype=int)
    prob = d[col].to_numpy(dtype=float)
    return {'AUROC':roc_auc_score(y,prob) if np.unique(y).size==2 else np.nan,
            'average_precision':average_precision_score(y,prob) if y.sum() else np.nan,
            'brier_score':brier_score_loss(y,prob)}


def main():
    for path in (PRED, TARGETS, RECEIPT):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')
    receipt = json.loads(RECEIPT.read_text(encoding='utf-8'))
    if receipt.get('frozen_manifest_files_checked') != 28:
        raise ValueError('Step 15 receipt was not verified; stop.')
    p = pd.read_csv(PRED)
    t = pd.read_csv(TARGETS, low_memory=False)
    required_p = {'Country','Year','partition','y',*SCORES}
    required_t = {'Country','Year','outcome_eligible','Incidence_per_million',
                  'prior_3y_median_per_million',
                  'surge_A_2x_1pm','surge_B_3x_1pm',MAIN}
    if required_p-set(p.columns) or required_t-set(t.columns):
        raise ValueError(f'Missing prediction columns: {sorted(required_p-set(p.columns))}; '
                         f'target columns: {sorted(required_t-set(t.columns))}')
    if p.duplicated(['partition','Country','Year']).any() or t.duplicated(['Country','Year']).any():
        raise ValueError('Duplicate prediction/target key detected.')
    if p['Country'].isna().any() or p['Year'].isna().any():
        raise ValueError('Missing prediction keys.')
    if set(p['partition']) != {'test','unseen_country_test'} or not p['Year'].between(2022,2025).all():
        raise ValueError('Unexpected test years or partitions.')
    if not p['y'].isin([0,1]).all(): raise ValueError('Invalid frozen labels.')
    if p[SCORES].isna().any().any(): raise ValueError('Missing frozen scores.')
    for s in MODEL_SCORES:
        if not p[s].between(0,1).all(): raise ValueError(f'Invalid probability score: {s}')
    if (p['partition'].value_counts().to_dict() != {'test':505,'unseen_country_test':119}):
        raise ValueError('Unexpected frozen test cohort sizes; do not silently proceed.')

    t['Year'] = pd.to_numeric(t['Year'], errors='raise').astype(int)
    t['eligible_bool'] = number_bool(t['outcome_eligible'])
    for c in ['Incidence_per_million','prior_3y_median_per_million',
              'surge_A_2x_1pm','surge_B_3x_1pm',MAIN]:
        t[c] = pd.to_numeric(t[c],errors='coerce')
    if t.loc[t['eligible_bool'],[MAIN,'Incidence_per_million','prior_3y_median_per_million']].isna().any().any():
        raise ValueError('Eligible target rows contain missing target/incidence/baseline.')
    if not t.loc[t['eligible_bool'], MAIN].isin([0,1]).all():
        raise ValueError('Invalid candidate C target values.')

    # Onset eligibility and label from *consecutive calendar-year* observed C labels.
    # An unknown previous-year C is neither a negative nor a confirmed onset.
    t = t.sort_values(['Country','Year']).reset_index(drop=True)
    grp = t.groupby('Country', sort=False)
    previous_year = grp['Year'].shift(1)
    previous_elig = grp['eligible_bool'].shift(1).eq(True)
    previous_C = grp[MAIN].shift(1)
    t['strict_onset_eligible'] = (t['eligible_bool'] &
        (t['Year']-previous_year).eq(1) & previous_elig & previous_C.isin([0,1]))
    t['strict_onset'] = np.where(t['strict_onset_eligible'],
        (t[MAIN].eq(1) & previous_C.eq(0)).astype(int), np.nan)

    t['C_positive_baseline'] = np.where(t['eligible_bool'] & t['prior_3y_median_per_million'].gt(0),
                                        t[MAIN],np.nan)
    incidence = t['Incidence_per_million']
    baseline = t['prior_3y_median_per_million']
    for threshold in (2.,10.):
        col = f'surge_2x_{int(threshold)}pm'
        label = incidence.ge(threshold) & incidence.ge(2*baseline) & (incidence-baseline).ge(0.5)
        t[col] = np.where(t['eligible_bool'], label.astype(int), np.nan)

    alt = {
        'C_primary_frozen': MAIN,
        'A_broad_2x_1pm':'surge_A_2x_1pm',
        'B_strict_3x_1pm':'surge_B_3x_1pm',
        'C_exclude_zero_median':'C_positive_baseline',
        'strict_new_C_onset':'strict_onset',
        'C_threshold_2pm':'surge_2x_2pm',
        'C_threshold_10pm':'surge_2x_10pm',
    }
    merge_cols=['Country','Year','eligible_bool','prior_3y_median_per_million',
                'strict_onset_eligible',*set(alt.values())]
    d=p.merge(t[merge_cols],on=['Country','Year'],how='left',validate='many_to_one',indicator=True)
    if not d['_merge'].eq('both').all() or not d['eligible_bool'].eq(True).all():
        raise ValueError('Frozen test predictions do not match eligible original target rows.')
    if not d[MAIN].eq(d['y']).all():
        raise ValueError('Frozen test labels do not match C labels in source. STOP: possible data drift.')
    if d[MAIN].sum() != 176 or d.groupby('partition')['y'].sum().to_dict() != {'test':143,'unseen_country_test':33}:
        raise ValueError('Unexpected primary outcome counts: STOP without editing frozen protocol.')

    OUT.mkdir(parents=True, exist_ok=True)
    summaries=[]; decisions=[]; classifs=[]; annual=[]; transitions=[]
    for name, target_col in alt.items():
        for partition, g in d.groupby('partition',sort=True):
            v=g[g[target_col].notna()].copy()
            v['target']=v[target_col].astype(int)
            summaries.append({'outcome':name,'partition':partition,'evaluated':len(v),
                              'excluded_from_frozen_cohort':len(g)-len(v),
                              'events':int(v.target.sum()),'non_events':len(v)-int(v.target.sum()),
                              'event_rate':v.target.mean() if len(v) else np.nan})
            # Distinguish changed classification from eligibility exclusions.
            transitions.append({'outcome':name,'partition':partition,
                    'C1_new1':int(((v.y==1)&(v.target==1)).sum()),
                    'C1_new0':int(((v.y==1)&(v.target==0)).sum()),
                    'C0_new1':int(((v.y==0)&(v.target==1)).sum()),
                    'C0_new0':int(((v.y==0)&(v.target==0)).sum()),
                    'rows_excluded':len(g)-len(v)})
            if not len(v): continue
            for score in SCORES:
                if score in MODEL_SCORES:
                    classifs.append({'outcome':name,'partition':partition,'model':score,
                                    'n':len(v),'events':int(v.target.sum()),**classification(v,score)})
                for cap in (10,20):
                    decisions.append({'outcome':name,'partition':partition,'approach':score,
                                      'capacity_pct':cap,**evaluate_annual(v,score,cap)})
                    for year, year_df in v.groupby('Year',sort=True):
                        annual.append({'outcome':name,'partition':partition,'Year':year,
                                       'approach':score,'capacity_pct':cap,
                                       **evaluate_annual(year_df,score,cap)})

    out_summary=pd.DataFrame(summaries)
    dec=pd.DataFrame(decisions)
    cdf=pd.DataFrame(classifs)
    ann=pd.DataFrame(annual)
    for fname,frame in [('outcome_cohort_summary.csv',out_summary),
                        ('outcome_transition_audit.csv',pd.DataFrame(transitions)),
                        ('decision_metrics_by_outcome.csv',dec),
                        ('classification_by_outcome.csv',cdf),
                        ('annual_decision_by_outcome.csv',ann)]:
        frame.to_csv(OUT/fname,index=False)
    notes = '''STEP 17: POST-TEST EXPLORATORY OUTCOME ROBUSTNESS ONLY
Frozen model predictions and frozen selection rule used; no models re-trained or tuned.
This analysis occurs AFTER the primary held-out tests were opened at Step 15.
Alternative labels must not be used to replace the primary endpoint or choose a new winner.
Each alternative endpoint is compared on its outcome-evaluable rows; denominator
and annual selection capacity may change when rows are excluded.
For strict onset, only rows with an eligible observed consecutive preceding year
are retained. Previous positive => continuation and target 0; previous negative
and present C-positive => target 1. Missing prior labels are excluded, not negative.
Zero-median exclusion removes those rows from numerator AND denominator.
A/B, and threshold-2/10 definitions are researcher-defined annual incidence surges,
NOT WHO-confirmed outbreaks. Equal annual selection capacity is within each endpoint's
retained retrospective evaluation cohort. This is not a prospective deployment simulation.
Vaccination national WUENIC is retrospectively revised; reporting is heterogeneous.
Classification Brier score for alternate labels is descriptive, NOT recalibration.
Country and annual shocks can produce dependence; no new independent validation implied.
Do not alter 2022-2025 test predictions, frozen_protocol.json, or original data.
'''
    (OUT/'robustness_notes.txt').write_text(notes,encoding='utf-8')
    print('\nSTEP 17: POST-TEST OUTCOME ROBUSTNESS (NO TRAINING / NO TUNING)\n')
    print(out_summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nAnnual Recall@10%, selected comparisons:')
    show=dec[(dec.capacity_pct==10)&dec.approach.isin(['coverage_shortfall_rank','rf_regularized','xgb_regularized'])]
    print(show[['outcome','partition','approach','selected','events','detected','recall']]
          .to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print(f'\nSaved: {OUT.relative_to(ROOT)}')
    print('NOTE: POST-TEST exploratory robustness; frozen predictions and protocol unchanged.')

if __name__=='__main__':
    main()
