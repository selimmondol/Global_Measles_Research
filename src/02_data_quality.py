"""Step 2: read-only quality audit of the reproducibly rebuilt global measles panel.

From the project root run:
    python src/02_data_quality.py

Required: data/processed/Measles_Master_Rebuilt.csv (from 01_data_preparation.py)
Optional: prior Step 4 CSV in data/processed/ to audit population-matching differences.
Outputs: CSV reports under results/data_quality/. No source or processed file is modified.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data' / 'processed' / 'Measles_Master_Rebuilt.csv'
REPORTS = ROOT / 'results' / 'data_quality'
REPORTS.mkdir(parents=True, exist_ok=True)


def save_csv(frame, filename):
    frame.to_csv(REPORTS / filename, index=False)


def as_number(series):
    return pd.to_numeric(series, errors='coerce')


if not SOURCE.is_file():
    raise SystemExit(f'Missing {SOURCE}. Run: python src/01_data_preparation.py')
df = pd.read_csv(SOURCE, low_memory=False)
required = ['Country', 'Year', 'Measles_reported_cases', 'ISO3',
            'MCV1_WUENIC_pct', 'MCV2_WUENIC_pct', 'Population_WB', 'Incidence_per_million']
absent = sorted(set(required) - set(df.columns))
if absent:
    raise SystemExit(f'Missing columns in rebuilt dataset: {absent}')

for col in ['Year', 'Measles_reported_cases', 'MCV1_WUENIC_pct',
            'MCV2_WUENIC_pct', 'Population_WB', 'Incidence_per_million']:
    df[col] = as_number(df[col])

# Counts describe reported cases, not true infections; blank does not mean zero.
cols = ['Measles_reported_cases', 'MCV1_WUENIC_pct', 'MCV2_WUENIC_pct',
        'Population_WB', 'Incidence_per_million', 'ISO3']
missing = pd.DataFrame([{
    'Field': c, 'Available': int(df[c].notna().sum()),
    'Missing': int(df[c].isna().sum()),
    'Missing_pct': round(100 * df[c].isna().mean(), 2),
} for c in cols])
save_csv(missing, 'missingness_by_variable.csv')

by_year = df.groupby('Year', dropna=False).agg(
    country_rows=('Country', 'size'),
    reported_cases=('Measles_reported_cases', 'count'),
    mcv1=('MCV1_WUENIC_pct', 'count'),
    mcv2=('MCV2_WUENIC_pct', 'count'),
    population=('Population_WB', 'count'),
    incidence=('Incidence_per_million', 'count'),
).reset_index()
by_year['complete_case_vax_population'] = (df[required[2:7]]
    .notna().all(axis=1).groupby(df['Year']).sum().reindex(by_year['Year']).to_numpy())
save_csv(by_year, 'coverage_by_year.csv')

by_country = df.groupby(['Country', 'ISO3'], dropna=False).agg(
    years=('Year', 'size'),
    years_cases=('Measles_reported_cases', 'count'),
    years_mcv1=('MCV1_WUENIC_pct', 'count'),
    years_mcv2=('MCV2_WUENIC_pct', 'count'),
    years_population=('Population_WB', 'count'),
    years_incidence=('Incidence_per_million', 'count'),
).reset_index().sort_values(['years_incidence', 'Country'])
save_csv(by_country, 'coverage_by_country.csv')

zero_cases = int(df['Measles_reported_cases'].eq(0).sum())
missing_cases = int(df['Measles_reported_cases'].isna().sum())
negative_cases = df['Measles_reported_cases'].lt(0)
noninteger_cases = (df['Measles_reported_cases'].notna() &
                    ~np.isclose(df['Measles_reported_cases'] % 1, 0))
invalid_vax = (df[['MCV1_WUENIC_pct','MCV2_WUENIC_pct']].lt(0) |
               df[['MCV1_WUENIC_pct','MCV2_WUENIC_pct']].gt(100)).any(axis=1)
invalid_pop = df['Population_WB'].notna() & df['Population_WB'].le(0)
no_iso = df['ISO3'].isna()
duplicate = df.duplicated(['Country', 'Year'], keep=False)
invalid_year = ~df['Year'].between(2000, 2025)

# Separate anomalies from true errors; surveillance totals and WUENIC estimates
# may be revised and cannot be independently certified by this internal check.
issues = []
for flag, description in [
    (negative_cases, 'Negative reported cases'),
    (noninteger_cases, 'Non-integer reported cases'),
    (invalid_vax, 'WUENIC vaccine coverage outside 0–100%'),
    (invalid_pop, 'Nonpositive population'),
    (duplicate, 'Duplicate country/year'),
    (invalid_year, 'Year outside 2000–2025'),
    (no_iso, 'No ISO3 match (potential unmatched territory)'),
]:
    for _, row in df.loc[flag, ['Country','Year','ISO3', 'Measles_reported_cases',
                                 'MCV1_WUENIC_pct','MCV2_WUENIC_pct','Population_WB']].iterrows():
        item = row.to_dict()
        item['Issue'] = description
        issues.append(item)
save_csv(pd.DataFrame(issues, columns=['Country','Year','ISO3','Measles_reported_cases',
                                       'MCV1_WUENIC_pct','MCV2_WUENIC_pct','Population_WB','Issue']),
         'flagged_records.csv')

# Recompute incidence from reported cases and population, for arithmetic consistency.
recalc = df['Measles_reported_cases'] / df['Population_WB'] * 1_000_000
both = recalc.notna() & df['Incidence_per_million'].notna()
incidence_mismatch = both & ~np.isclose(recalc, df['Incidence_per_million'], rtol=1e-8, atol=1e-8)
save_csv(df.loc[incidence_mismatch, ['Country','Year','Measles_reported_cases',
                                    'Population_WB','Incidence_per_million']], 'incidence_mismatches.csv')

# Optional: compare only the ability to calculate incidence. This does NOT replace
# scientific source verification or silently choose which population is correct.
old_candidates = [ROOT / 'data' / 'processed' / name for name in (
    'Measles_Step4_Population_Adjusted_Data.csv',
    'Measles_Step4_Population_Adjusted.csv',
)]
previous = next((p for p in old_candidates if p.is_file()), None)
compare_summary = 'Not compared (optional prior Step 4 CSV not found)'
if previous:
    old = pd.read_csv(previous, usecols=['Country','Year','Incidence_per_million'])
    old['Incidence_per_million'] = as_number(old['Incidence_per_million'])
    old = old.rename(columns={'Incidence_per_million':'Old_incidence'})
    merged = df[['Country','Year','ISO3','Population_WB','Incidence_per_million']].merge(
        old, on=['Country','Year'], how='outer', validate='one_to_one', indicator=True)
    current_available = merged['Incidence_per_million'].notna()
    prior_available = merged['Old_incidence'].notna()
    changes = merged.loc[current_available.ne(prior_available), [
        'Country','Year','ISO3','Population_WB','Incidence_per_million','Old_incidence']].copy()
    changes['Change_type'] = np.where(changes['Incidence_per_million'].notna(),
                                      'Only rebuilt available', 'Only previous Step 4 available')
    save_csv(changes, 'step4_incidence_availability_differences.csv')
    unequal = (current_available & prior_available &
               ~np.isclose(merged['Incidence_per_million'], merged['Old_incidence'], rtol=1e-6, atol=1e-7))
    save_csv(merged.loc[unequal, ['Country','Year','ISO3','Incidence_per_million','Old_incidence']],
             'step4_incidence_value_differences.csv')
    compare_summary = (f'Previous Step 4 compared; {len(changes)} availability differences, '
                       f'{int(unequal.sum())} value differences')

summary = pd.DataFrame([
    ['Countries/territories', df['Country'].nunique()],
    ['Country-year records', len(df)],
    ['Zero reported cases (observed)', zero_cases],
    ['Missing reported cases (not zero)', missing_cases],
    ['Negative reported case records', int(negative_cases.sum())],
    ['Noninteger reported case records', int(noninteger_cases.sum())],
    ['Vaccination records outside 0–100%', int(invalid_vax.sum())],
    ['Population <= 0 records', int(invalid_pop.sum())],
    ['Missing ISO3 records', int(no_iso.sum())],
    ['Duplicate country-year records', int(duplicate.sum())],
    ['Recalculated incidence inconsistencies', int(incidence_mismatch.sum())],
    ['Previous Step 4 comparison', compare_summary],
], columns=['Check', 'Result'])
save_csv(summary, 'data_quality_summary.csv')
print('\nDATA QUALITY AUDIT (read only)\n')
print(summary.to_string(index=False))
print(f'\nReports saved in: {REPORTS.relative_to(ROOT)}/')
print('NOTE: Matching and arithmetic checks do NOT independently verify WHO surveillance accuracy.')
