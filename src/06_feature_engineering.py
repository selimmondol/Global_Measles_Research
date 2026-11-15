#!/usr/bin/env python3
"""Step 6: leakage-conscious, reproducible feature engineering for global measles.

Run from the Measles project root:
    python src/06_feature_engineering.py

Input: data/processed/Measles_Resurgence_Candidates.csv
Outputs:
    data/processed/Measles_Model_Features.csv
    results/feature_engineering/feature_audit.csv
    results/feature_engineering/feature_dictionary.csv
    results/feature_engineering/feature_missingness.csv
    results/feature_engineering/methodology_notes.txt

No model is fitted and no original files are changed. Candidate outcome labels are
NOT WHO-confirmed outbreak labels. Feature information for target year t is taken
ONLY from t-1 or earlier. The WHO/WUENIC historical estimates used here are
CURRENT REVISIONS, not proof of historical real-time availability.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data' / 'processed' / 'Measles_Resurgence_Candidates.csv'
OUTPUT = ROOT / 'data' / 'processed' / 'Measles_Model_Features.csv'
REPORT = ROOT / 'results' / 'feature_engineering'
TARGETS = ['surge_A_2x_1pm','surge_B_3x_1pm','surge_C_2x_5pm']
RAW = ['Incidence_per_million','MCV1_WUENIC_pct','MCV2_WUENIC_pct','Population_WB']
REQUIRED = ['Country','Year', 'outcome_eligible', *TARGETS, *RAW]


def main():
    if not INPUT.exists():
        raise SystemExit(f'Missing input: {INPUT}\nRun python src/03_resurgence_definition.py first.')
    df = pd.read_csv(INPUT, low_memory=False)
    missing = sorted(set(REQUIRED)-set(df.columns))
    if missing:
        raise SystemExit(f'Missing columns: {missing}')
    df['Year'] = pd.to_numeric(df['Year'], errors='raise').astype(int)
    if df[['Country','Year']].isna().any().any() or df.duplicated(['Country','Year']).any():
        raise SystemExit('Missing or duplicate country-year keys; fix upstream.')
    if not df['Year'].between(2000, 2025).all():
        raise SystemExit('Unexpected years; check source data.')
    for c in RAW+TARGETS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.sort_values(['Country','Year']).reset_index(drop=True)
    eligible = df['outcome_eligible'].astype('boolean').fillna(False)
    for t in TARGETS:
        if df.loc[~eligible, t].notna().any() or not df.loc[eligible, t].isin([0,1]).all():
            raise SystemExit(f'Eligibility/target inconsistency for {t}')

    g = df.groupby('Country', sort=False)
    # Calendar-aware shifts: a missing calendar year never becomes an artificial lag.
    lag_sources = [('Incidence_per_million', 'incidence'),
                   ('MCV1_WUENIC_pct', 'mcv1'),
                   ('MCV2_WUENIC_pct', 'mcv2'),
                   ('Population_WB', 'population')]
    for src, prefix in lag_sources:
        for k in (1,2,3):
            years_lagged = g['Year'].shift(k)
            values_lagged = g[src].shift(k)
            df[f'{prefix}_lag{k}'] = values_lagged.where((df['Year']-years_lagged).eq(k))

    inc_cols = [f'incidence_lag{k}' for k in (1,2,3)]
    complete_history = df[inc_cols].notna().all(axis=1)
    df['incidence_history_3y_complete'] = complete_history.astype('int8')
    df['incidence_median_3y'] = df[inc_cols].median(axis=1).where(complete_history)
    df['incidence_mean_3y'] = df[inc_cols].mean(axis=1).where(complete_history)
    df['incidence_std_3y'] = df[inc_cols].std(axis=1, ddof=0).where(complete_history)
    df['incidence_max_3y'] = df[inc_cols].max(axis=1).where(complete_history)
    df['incidence_change_1y'] = df['incidence_lag1'] - df['incidence_lag2']
    df['incidence_history_zero_median'] = pd.Series(
        np.where(complete_history, df['incidence_median_3y'].eq(0).astype(float), np.nan))

    for v in ('mcv1','mcv2'):
        df[f'{v}_shortfall_lag1'] = (100 - df[f'{v}_lag1']).clip(lower=0).where(df[f'{v}_lag1'].notna())
        df[f'{v}_change_prior'] = df[f'{v}_lag1'] - df[f'{v}_lag2']
        df[f'{v}_lag1_missing'] = df[f'{v}_lag1'].isna().astype('int8')
    df['dose_coverage_gap_lag1'] = (df['mcv1_lag1']-df['mcv2_lag1']).where(
        df[['mcv1_lag1','mcv2_lag1']].notna().all(axis=1))
    df['both_doses_ge95_lag1'] = pd.Series(np.where(
        df[['mcv1_lag1','mcv2_lag1']].notna().all(axis=1),
        (df['mcv1_lag1'] >= 95) & (df['mcv2_lag1'] >= 95), np.nan), dtype='float64')
    df['population_log_lag1'] = np.log(df['population_lag1'].where(df['population_lag1'] > 0))
    df['incidence_lag1_missing'] = df['incidence_lag1'].isna().astype('int8')

    # This is a predictive data table, NOT a trained model dataset yet.
    # Primary candidate C is not a validated WHO outbreak endpoint.
    features = [
        'incidence_lag1','incidence_lag2','incidence_lag3',
        'incidence_median_3y','incidence_mean_3y','incidence_std_3y',
        'incidence_max_3y','incidence_change_1y','incidence_history_zero_median',
        'mcv1_lag1','mcv1_lag2','mcv2_lag1','mcv2_lag2',
        'mcv1_shortfall_lag1','mcv2_shortfall_lag1',
        'mcv1_change_prior','mcv2_change_prior','dose_coverage_gap_lag1',
        'both_doses_ge95_lag1','population_log_lag1',
        'mcv1_lag1_missing','mcv2_lag1_missing','incidence_lag1_missing',
        'incidence_history_3y_complete'
    ]
    meta = ['Country','Year','ISO3'] if 'ISO3' in df.columns else ['Country','Year']
    output_cols = meta + ['outcome_eligible'] + TARGETS + features
    out = df[output_cols].copy()
    # Keep missing target labels missing; do not silently classify them as 0.
    assert len(out) == len(df)
    assert not out.duplicated(['Country','Year']).any()
    assert out.loc[~eligible, TARGETS].isna().all().all()
    # Independently reconstruct a subset of time-aware lags from chronological source.
    prior_check = g['Incidence_per_million'].shift(1).where((df['Year']-g['Year'].shift(1)).eq(1))
    pd.testing.assert_series_equal(out['incidence_lag1'], prior_check, check_names=False)
    # Explicitly never write target-year incidence/cases/coverage to model features.
    assert not set(RAW + ['Measles_reported_cases', 'prior_3y_median_per_million']).intersection(features)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    n_eligible = int(eligible.sum())
    complete_vacc = out[['mcv1_lag1','mcv2_lag1']].notna().all(axis=1)
    audit = [
        ('All country-year rows',len(out)),
        ('Eligible candidate outcomes',n_eligible),
        ('Candidate C positive outcome years',int(out['surge_C_2x_5pm'].eq(1).sum())),
        ('Eligible with both prior-year vaccines',int((eligible & complete_vacc).sum())),
        ('Eligible with complete 3-year incidence history',int((eligible & complete_history).sum())),
        ('Eligible with prior-year population',int((eligible & out['population_log_lag1'].notna()).sum())),
        ('Unique countries/territories',int(out['Country'].nunique())),
        ('Duplicate country-year rows',int(out.duplicated(['Country','Year']).sum()))
    ]
    pd.DataFrame(audit, columns=['Metric','Value']).to_csv(REPORT/'feature_audit.csv', index=False)
    pd.DataFrame([{'Feature':c, 'Missing_all_rows':int(out[c].isna().sum()),
                   'Missing_eligible_rows':int(out.loc[eligible,c].isna().sum()),
                   'Available_eligible_rows':int(out.loc[eligible,c].notna().sum())}
                  for c in features]).to_csv(REPORT/'feature_missingness.csv', index=False)
    meanings = {
      'incidence_lag1':'Reported incidence per million at t-1',
      'incidence_lag2':'Reported incidence per million at t-2',
      'incidence_lag3':'Reported incidence per million at t-3',
      'incidence_median_3y':'Median of t-1, t-2 and t-3 incidence; complete history only',
      'incidence_mean_3y':'Mean of the three prior annual incidences',
      'incidence_std_3y':'Population standard deviation of the three prior annual incidences',
      'incidence_max_3y':'Maximum incidence among t-1 through t-3',
      'incidence_change_1y':'Incidence(t-1) minus incidence(t-2)',
      'incidence_history_zero_median':'Whether the three-year median is zero, if observed',
      'mcv1_lag1':'Current-revision WUENIC MCV1 coverage (%) assigned to t-1',
      'mcv1_lag2':'Current-revision WUENIC MCV1 coverage (%) assigned to t-2',
      'mcv2_lag1':'Current-revision WUENIC MCV2 coverage (%) assigned to t-1',
      'mcv2_lag2':'Current-revision WUENIC MCV2 coverage (%) assigned to t-2',
      'mcv1_shortfall_lag1':'100 minus MCV1 coverage at t-1 (percentage points)',
      'mcv2_shortfall_lag1':'100 minus MCV2 coverage at t-1 (percentage points)',
      'mcv1_change_prior':'MCV1(t-1) minus MCV1(t-2)',
      'mcv2_change_prior':'MCV2(t-1) minus MCV2(t-2)',
      'dose_coverage_gap_lag1':'MCV1(t-1) minus MCV2(t-1)',
      'both_doses_ge95_lag1':'1 if both preceding-year national coverage estimates >=95%',
      'population_log_lag1':'Natural log of WB population at t-1',
      'mcv1_lag1_missing':'1 if prior-year MCV1 unavailable',
      'mcv2_lag1_missing':'1 if prior-year MCV2 unavailable',
      'incidence_lag1_missing':'1 if prior-year incidence unavailable',
      'incidence_history_3y_complete':'1 if three consecutive prior incidence years observed',
    }
    pd.DataFrame([{'Feature':c,'Definition':meanings[c],'Permitted_temporal_window':'t-1 or earlier'}
                  for c in features]).to_csv(REPORT/'feature_dictionary.csv',index=False)
    (REPORT/'methodology_notes.txt').write_text(
        'FEATURE ENGINEERING AUDIT (NO MODEL TRAINING)\n'
        'Outcome: researcher-defined annual reported-incidence surge candidates, NOT WHO-confirmed outbreaks.\n'
        'For predicting target-year t, all predictive features originate from t-1 or earlier.\n'
        'Historical WUENIC values are current-revision estimates; historical real-time vintages are unverified.\n'
        'NO scaling, imputation, encoding or feature selection fitted across the full dataset.\n'
        'Country/Year/ISO3 are identifiers and NOT part of the numeric feature list.\n'
        'Current-year measles cases, incidence, vaccination coverage and population are excluded from predictive features.\n'
        'Some features (historical incidence median) are also used in the researcher-defined target rule.\n'
        'This is not future-outcome leakage but may favor the historical-incidence baseline; benchmark fairly.\n'
        'Missing values were preserved. Models must fit imputers and scalers on training folds only.\n'
        'Do not interpret national vaccination coverage as individual immunity.\n'
        'Pre-specify time splits and held-out-country testing BEFORE fitting any models.\n', encoding='utf-8'
    )
    print('\nSTEP 6: FEATURE ENGINEERING (no model training)\n')
    for k,v in audit:
        print(f'{k:<51}{v:>8}')
    print(f'\nSaved: {OUTPUT.relative_to(ROOT)}')
    print(f'Reports: {REPORT.relative_to(ROOT)}/')
    print('NOTE: Labels remain provisional; missingness preserved; no imputation or scaling.')

if __name__ == '__main__':
    main()
