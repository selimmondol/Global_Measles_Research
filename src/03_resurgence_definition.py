"""Step 3: reproducible candidate measles resurgence outcomes (NOT WHO outbreak labels).

Run from the Measles project root:
    python src/03_resurgence_definition.py

Reads:  data/processed/Measles_Master_Rebuilt.csv
Writes: data/processed/Measles_Resurgence_Candidates.csv
        results/resurgence_definition/*.csv
No original WHO, World Bank, or rebuilt master datasets are modified.

This is an exploratory outcome audit, not model training. The definitions below
are sensitivity candidates, not internationally validated outbreak definitions.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data' / 'processed' / 'Measles_Master_Rebuilt.csv'
OUT = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'results' / 'resurgence_definition'
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

if not SOURCE.is_file():
    raise SystemExit(f'Missing input: {SOURCE}\nFirst run python src/01_data_preparation.py')

df = pd.read_csv(SOURCE, low_memory=False)
required = ['Country', 'Year', 'Measles_reported_cases', 'Population_WB',
            'Incidence_per_million', 'MCV1_WUENIC_pct', 'MCV2_WUENIC_pct']
missing_cols = sorted(set(required) - set(df.columns))
if missing_cols:
    raise SystemExit(f'Input is missing required columns: {missing_cols}')
for col in required[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
if df[['Country', 'Year']].isna().any().any() or df.duplicated(['Country', 'Year']).any():
    raise SystemExit('Missing/duplicate Country-Year keys: correct upstream preparation first.')
if not df['Year'].between(2000, 2025).all():
    raise SystemExit('Unexpected year range in input.')
df = df.sort_values(['Country','Year']).reset_index(drop=True)
if (df['Incidence_per_million'].dropna() < 0).any():
    raise SystemExit('Negative incidence detected: correct source first.')

# Require ALL THREE *consecutive* preceding calendar years with observed
# incidence. Missing reporting is never interpreted as zero cases.
grp = df.groupby('Country', sort=False)
for k in (1, 2, 3):
    y = grp['Year'].shift(k)
    x = grp['Incidence_per_million'].shift(k)
    df[f'incidence_lag{k}'] = x.where((df['Year'] - y).eq(k))

history_cols = ['incidence_lag1','incidence_lag2','incidence_lag3']
df['history_3y_complete'] = df[history_cols].notna().all(axis=1)
df['prior_3y_median_per_million'] = df[history_cols].median(axis=1).where(df['history_3y_complete'])
df['outcome_eligible'] = df['Incidence_per_million'].notna() & df['history_3y_complete']

# Candidate incidence-based growth definitions. Ratios use inequalities rather
# than divisions, so prior median = 0 does not create infinite ratios.
# For primary exploratory candidate: incidence >=1/million AND >= 2x prior
# 3-year median AND absolute increase >=0.5/million.
def candidate(name, ratio, minimum_incidence, minimum_increase):
    x = df['Incidence_per_million']
    baseline = df['prior_3y_median_per_million']
    flag = (x >= minimum_incidence) & (x >= ratio * baseline) & \
           ((x - baseline) >= minimum_increase)
    df[name] = pd.Series(np.where(df['outcome_eligible'], flag.astype(int), np.nan),
                         index=df.index, dtype='float64')

candidate('surge_A_2x_1pm', 2, 1.0, 0.5)
candidate('surge_B_3x_1pm', 3, 1.0, 0.5)
candidate('surge_C_2x_5pm', 2, 5.0, 0.5)
candidate('surge_D_2x_5pm_2increase', 2, 5.0, 2.0)

# Prior-year WUENIC estimates are used as retrospective indicators. They are
# NOT contemporaneous predictors and are NOT proof of historical availability.
for col in ('MCV1_WUENIC_pct', 'MCV2_WUENIC_pct'):
    shifted_year = grp['Year'].shift(1)
    df[f'prior_{col}'] = grp[col].shift(1).where((df['Year'] - shifted_year).eq(1))
df['prior_both_mcv_available'] = df[['prior_MCV1_WUENIC_pct',
                                      'prior_MCV2_WUENIC_pct']].notna().all(axis=1)
df['prior_both_mcv_ge95'] = pd.Series(
    np.where(df['prior_both_mcv_available'],
             df['prior_MCV1_WUENIC_pct'].ge(95) & df['prior_MCV2_WUENIC_pct'].ge(95),
             np.nan), index=df.index, dtype='float64')

# Report denominators. Vaccine subgroup counts use only outcome-eligible rows
# with observed prior-year MCV1 and MCV2 (missing is never counted as low coverage).
def summarize(frame, group_label):
    entry = {'Group': group_label,
             'Country_year_rows': len(frame),
             'Outcome_eligible': int(frame['outcome_eligible'].sum()),
             'Outcome_ineligible': int((~frame['outcome_eligible']).sum())}
    for col in ['surge_A_2x_1pm','surge_B_3x_1pm',
                'surge_C_2x_5pm','surge_D_2x_5pm_2increase']:
        elig = frame[col].notna()
        entry[f'{col}_eligible'] = int(elig.sum())
        entry[f'{col}_events'] = int(frame[col].eq(1).sum())
        entry[f'{col}_event_rate_pct'] = round(100 * frame[col].eq(1).sum() / elig.sum(), 2) if elig.sum() else np.nan
    return entry

all_summary = [summarize(df, 'All country-years')]
with_prior = df[df['prior_both_mcv_available']]
all_summary.append(summarize(with_prior, 'Prior-year MCV1 & MCV2 observed'))
high = with_prior[with_prior['prior_both_mcv_ge95'].eq(1)]
all_summary.append(summarize(high, 'Prior-year MCV1 & MCV2 >=95%'))
pd.DataFrame(all_summary).to_csv(REPORTS / 'definition_summary.csv', index=False)

by_year = pd.DataFrame([summarize(g, str(int(year))) for year, g in df.groupby('Year')])
by_year.to_csv(REPORTS / 'events_by_year.csv', index=False)
by_country = pd.DataFrame([summarize(g, name) for name, g in df.groupby('Country')])
by_country = by_country.rename(columns={'Group': 'Country'})
by_country.to_csv(REPORTS / 'events_by_country.csv', index=False)

# Flag eligible years for which vaccination coverage is absent rather than
# dropping them silently. This helps later sample-selection assessment.
exclusions = pd.DataFrame([
    {'Reason': 'Current-year incidence missing', 'Rows': int(df['Incidence_per_million'].isna().sum())},
    {'Reason': 'Fewer than 3 consecutive prior years with incidence', 'Rows': int((~df['history_3y_complete']).sum())},
    {'Reason': 'Outcome eligible', 'Rows': int(df['outcome_eligible'].sum())},
    {'Reason': 'Outcome eligible but prior-year vaccination incomplete',
     'Rows': int((df['outcome_eligible'] & ~df['prior_both_mcv_available']).sum())},
])
exclusions.to_csv(REPORTS / 'eligibility_audit.csv', index=False)

out_path = OUT / 'Measles_Resurgence_Candidates.csv'
df.to_csv(out_path, index=False)
print('\nSTEP 3: CANDIDATE RESURGENCE AUDIT (not official outbreak labels)\n')
print(pd.DataFrame(all_summary).to_string(index=False))
print('\nEligibility audit:\n', exclusions.to_string(index=False))
print(f'\nSaved: {out_path.relative_to(ROOT)}')
print(f'Reports: {REPORTS.relative_to(ROOT)}/')
print('NOTE: Thresholds are provisional; population-adjusted reported-incidence surges are NOT confirmed outbreaks.')
print('NOTE: WHO/WUENIC current-revision historical estimates are NOT proven real-time data vintages.')
