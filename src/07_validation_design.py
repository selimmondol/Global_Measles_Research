#!/usr/bin/env python3
"""Step 7: Pre-modelling temporal and country-held-out validation audit.

Run from the Measles project root:
    python src/07_validation_design.py

Input: data/processed/Measles_Model_Features.csv
Outputs: results/validation_design/{split_summary.csv,yearly_event_counts.csv,
    split_assignments.csv,country_holdout.csv,coverage_by_split.csv,
    overlap_checks.csv,validation_design_notes.txt}

READ ONLY for input; no model fitting, imputation, scaling, or target changes.
The primary candidate outcome is a researcher-defined incidence surge, NOT a
WHO-confirmed outbreak. WUENIC revisions are not historical data vintages.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data' / 'processed' / 'Measles_Model_Features.csv'
REPORT = ROOT / 'results' / 'validation_design'
TARGET = 'surge_C_2x_5pm'
SEED = 'measles-country-holdout-v1-fixed-before-modeling'
HOLDOUT_SHARE = 0.20


def time_period(year):
    if 2003 <= year <= 2018:
        return 'train'
    if 2019 <= year <= 2021:
        return 'validation'
    if 2022 <= year <= 2025:
        return 'test'
    return 'prehistory'


def main():
    if not INPUT.exists():
        raise SystemExit(f'Missing {INPUT}; first run python src/06_feature_engineering.py')
    df = pd.read_csv(INPUT, low_memory=False)
    required = {'Country','Year','outcome_eligible',TARGET,'mcv1_lag1','mcv2_lag1',
                'incidence_lag1','incidence_lag2','incidence_lag3'}
    missing = sorted(required-set(df.columns))
    if missing:
        raise SystemExit(f'Missing expected columns: {missing}')
    df['Year'] = pd.to_numeric(df['Year'], errors='raise').astype(int)
    if df[['Country','Year']].isna().any().any() or df.duplicated(['Country','Year']).any():
        raise SystemExit('Invalid country-year keys: audit upstream inputs first')
    if not df['Year'].between(2000,2025).all():
        raise SystemExit('Unexpected year values')
    eligible = df['outcome_eligible'].astype('boolean').fillna(False)
    target = pd.to_numeric(df[TARGET], errors='coerce')
    if target.loc[~eligible].notna().any() or not target.loc[eligible].isin([0,1]).all():
        raise SystemExit('Target/eligibility inconsistency')
    if len(df) == 0 or eligible.sum() == 0:
        raise SystemExit('No eligible observations')
    df['eligible_for_outcome'] = eligible
    df['split_period'] = df['Year'].map(time_period)

    # Reserve countries before modelling using a documented deterministic hash.
    # Country count is based ONLY on whether there are eligible observations,
    # never on cases, event rates, vaccination, or model outcomes.
    eligible_countries = sorted(df.loc[eligible,'Country'].unique())
    ranked = sorted(eligible_countries,
                    key=lambda c: (hashlib.sha256(f'{SEED}|{c}'.encode()).hexdigest(), c))
    n_held = max(1, round(len(ranked)*HOLDOUT_SHARE))
    held = set(ranked[:n_held])
    df['country_holdout'] = df['Country'].isin(held)
    # Strictly separate fitting countries from unseen countries in ALL years.
    df['partition'] = np.select(
        [df['split_period'].eq('prehistory'), df['country_holdout'],
         df['split_period'].eq('train'),df['split_period'].eq('validation'),
         df['split_period'].eq('test')],
        ['prehistory','reserved_country','train','validation','test'],default='unassigned')
    # Assign reserved countries' earlier years as quarantined (not for training).
    df['evaluation_role'] = df['partition']
    df.loc[df['country_holdout'] & df['split_period'].eq('test'), 'evaluation_role'] = 'unseen_country_test'
    df.loc[df['country_holdout'] & ~df['split_period'].eq('test') & ~df['split_period'].eq('prehistory'), 'evaluation_role'] = 'reserved_not_used'

    df['vaccines_both_observed'] = df[['mcv1_lag1','mcv2_lag1']].notna().all(axis=1)
    df['incidence_lags_complete'] = df[['incidence_lag1','incidence_lag2','incidence_lag3']].notna().all(axis=1)
    df['target_is_positive'] = target.eq(1).fillna(False)
    df['high_vaccination_prior_year'] = df['vaccines_both_observed'] & df['mcv1_lag1'].ge(95) & df['mcv2_lag1'].ge(95)

    rows = []
    for name in ['train','validation','test','unseen_country_test','reserved_not_used','prehistory']:
        sub = df.loc[df['evaluation_role'].eq(name)]
        el = sub['eligible_for_outcome']
        e = sub.loc[el]
        positives = int(e['target_is_positive'].sum())
        rows.append({'split':name, 'calendar_years':f'{sub.Year.min()}–{sub.Year.max()}' if len(sub) else '',
                     'country_count':int(sub.Country.nunique()),'all_rows':len(sub),
                     'eligible_rows':len(e),'positive_events':positives,
                     'negative_events':int(len(e)-positives),
                     'event_rate_pct':round(100*positives/len(e),2) if len(e) else np.nan,
                     'both_vaccines_available_eligible':int(e.vaccines_both_observed.sum()),
                     'both_vaccines_missing_or_partial_eligible':int((~e.vaccines_both_observed).sum()),
                     'complete_incidence_lags_eligible':int(e.incidence_lags_complete.sum()),
                     'high_vaccination_eligible':int(e.high_vaccination_prior_year.sum()),
                     'high_vaccination_positive':int((e.high_vaccination_prior_year & e.target_is_positive).sum())})
    summary = pd.DataFrame(rows)

    years=[]
    for (period, year), sub in df.groupby(['evaluation_role','Year'],sort=True):
        e = sub.loc[sub.eligible_for_outcome]
        years.append({'split':period, 'year':year, 'countries_total':sub.Country.nunique(),
                      'eligible_rows':len(e),'positive_events':int(e.target_is_positive.sum()),
                      'both_vaccines_observed_eligible':int(e.vaccines_both_observed.sum())})
    yearly = pd.DataFrame(years)
    country = pd.DataFrame({'Country':ranked,
                            'holdout_status':['unseen_country' if c in held else 'development_country' for c in ranked],
                            'eligible_years':[int((df.Country.eq(c)&eligible).sum()) for c in ranked]})
    active = {name:set(df.loc[df.evaluation_role.eq(name)&eligible,'Country'])
              for name in ('train','validation','test','unseen_country_test')}
    overlap=[]
    for left,right in [('train','validation'),('train','test'),('train','unseen_country_test'),
                       ('validation','test'),('validation','unseen_country_test'),
                       ('test','unseen_country_test')]:
        common=active[left]&active[right]
        overlap.append({'split_1':left,'split_2':right,'country_overlap_count':len(common),
                        'design_expectation':'ZERO (unseen country)' if 'unseen_country_test' in (left,right) else 'Allowed (temporal split)'})
    if any((active[x]&active['unseen_country_test']) for x in ('train','validation','test')):
        raise AssertionError('Country-held-out group leaked into development samples')
    if not df.loc[df.evaluation_role.eq('train'),'Year'].between(2003,2018).all():
        raise AssertionError('Training time range invalid')
    if not df.loc[df.evaluation_role.eq('validation'),'Year'].between(2019,2021).all():
        raise AssertionError('Validation time range invalid')
    if not df.loc[df.evaluation_role.eq('test'),'Year'].between(2022,2025).all():
        raise AssertionError('Test time range invalid')
    if not df.loc[df.evaluation_role.eq('unseen_country_test'),'Year'].between(2022,2025).all():
        raise AssertionError('Heldout-country test time range invalid')

    REPORT.mkdir(parents=True,exist_ok=True)
    summary.to_csv(REPORT/'split_summary.csv',index=False)
    yearly.to_csv(REPORT/'yearly_event_counts.csv',index=False)
    country.to_csv(REPORT/'country_holdout.csv',index=False)
    pd.DataFrame(overlap).to_csv(REPORT/'overlap_checks.csv',index=False)
    df[['Country','Year','eligible_for_outcome','split_period','country_holdout',
        'evaluation_role','vaccines_both_observed','incidence_lags_complete']].to_csv(
        REPORT/'split_assignments.csv',index=False)
    summary[['split','eligible_rows','both_vaccines_available_eligible',
             'both_vaccines_missing_or_partial_eligible','high_vaccination_eligible']].to_csv(
        REPORT/'coverage_by_split.csv',index=False)
    (REPORT/'validation_design_notes.txt').write_text(
      'STEP 7: PROVISIONAL VALIDATION DESIGN (NO MODEL TRAINING)\n'
      'Target is researcher-defined annual reported-incidence surge, NOT WHO-confirmed outbreak.\n'
      'Training 2003-2018; tuning/validation 2019-2021; locked future test 2022-2025.\n'
      'A deterministic 20% of countries with eligible history are reserved from ALL model development.\n'
      'Their 2022-2025 eligible outcomes form a separate unseen-country future-year evaluation.\n'
      'Holdout selection: SHA256 of fixed salt + exact country name; sorted; first rounded 20%.\n'
      f'Salt: {SEED}; reserved {n_held} of {len(ranked)} countries with eligible labels.\n'
      'Never tune models, thresholds, preprocessing or feature selection on either test partition.\n'
      'Train-only imputation/scaling/encoding; fit calibration on validation or training folds only.\n'
      'Annual labels and serial observations are dependent within country; report country-cluster uncertainty.\n'
      '2020-2021 surveillance and vaccination disruptions warrant sensitivity analyses.\n'
      'Current-revision WUENIC values are not proven to have been available in historical real time.\n'
      'Resurgence definition is provisional; lock protocol and time splits before fitting models.\n'
      'Pre-specified country holdout checks geographical transfer, NOT fully independent contemporaneous\n'
      'global shocks or true cross-border epidemiological transportability.\n'
      'No predictor values were imputed and no modelling performed.\n',encoding='utf-8')

    print('\nSTEP 7: VALIDATION DESIGN (NO MODEL TRAINING)\n')
    print(summary.to_string(index=False))
    print(f'\nHoldout countries: {n_held} of {len(ranked)} eligible countries (20% predetermined hash)')
    print('Confirmed: unseen-country test countries absent from all development partitions')
    print(f'Reports saved: {REPORT.relative_to(ROOT)}/')
    print('NOTE: This is a proposal to lock before modelling; current-revision WUENIC data are retrospective.')

if __name__ == '__main__':
    main()
