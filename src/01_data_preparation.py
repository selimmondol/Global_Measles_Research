"""Rebuild the Global Measles country-year dataset from unmodified WHO and World Bank files.

Run from the Measles project root:
    python src/01_data_preparation.py

Input files are discovered by filename prefixes inside data/raw/.
Outputs go to data/processed/ and results/; raw files are never changed.
"""
from pathlib import Path
from zipfile import ZipFile
import io
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def find_one(prefix, suffix):
    matches = sorted(p for p in RAW.iterdir() if p.name.lower().startswith(prefix.lower()) and p.name.lower().endswith(suffix))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one {prefix}*{suffix} in {RAW}; found {len(matches)}: {[p.name for p in matches]}')
    return matches[0]

cases_file = find_one('Measles reported cases and incidence', '.xlsx')
vax_file = find_one('Measles vaccination coverage', '.xlsx')
pop_file = find_one('API_SP.POP.TOTL', '.zip')

# 1. WHO annual reported measles cases; ignore the export metadata row.
cases = pd.read_excel(cases_file, sheet_name='Sheet1')
cases = cases.loc[cases['Disease'].eq('Measles')].copy()
year_columns = [c for c in cases.columns if str(c).isdigit() and 2000 <= int(c) <= 2025]
assert len(year_columns) == 26, f'Expected 26 years, found {len(year_columns)}'
cases = cases.melt(id_vars=['Country / Region'], value_vars=year_columns,
                   var_name='Year', value_name='Measles_reported_cases')
cases = cases.rename(columns={'Country / Region': 'Country'})
cases['Year'] = cases['Year'].astype(int)
cases['Measles_reported_cases'] = pd.to_numeric(
    cases['Measles_reported_cases'].astype('string').str.replace(',', '', regex=False), errors='coerce'
)
assert not cases.duplicated(['Country', 'Year']).any(), 'Duplicate case country-year!'

# 2. Only WUENIC estimates: never blend ADMIN / OFFICIAL into primary estimates.
vax = pd.read_excel(vax_file, sheet_name='Sheet1')
vax = vax.loc[(vax['COVERAGE_CATEGORY'] == 'WUENIC') & vax['ANTIGEN'].isin(['MCV1', 'MCV2'])].copy()
vax['YEAR'] = pd.to_numeric(vax['YEAR'], errors='coerce')
vax = vax.loc[vax['YEAR'].between(2000, 2025)]
vax['COVERAGE'] = pd.to_numeric(vax['COVERAGE'], errors='coerce')
assert not vax.duplicated(['NAME', 'YEAR', 'ANTIGEN']).any(), 'Duplicate WUENIC country-year-antigen!'
vax = vax.pivot(index=['NAME', 'YEAR', 'CODE'], columns='ANTIGEN', values='COVERAGE').reset_index()
vax.columns.name = None
vax = vax.rename(columns={'NAME': 'Country', 'YEAR': 'Year', 'CODE': 'ISO3',
                          'MCV1': 'MCV1_WUENIC_pct', 'MCV2': 'MCV2_WUENIC_pct'})

# 3. World Bank population from the original ZIP; avoid regional aggregate rows
# by linking to WHO vaccination ISO3 codes, not names wherever possible.
with ZipFile(pop_file) as z:
    csv_names = [x for x in z.namelist() if x.startswith('API_') and x.endswith('.csv')]
    if len(csv_names) != 1:
        raise ValueError('Population ZIP must contain exactly one main API_*.csv')
    population = pd.read_csv(io.BytesIO(z.read(csv_names[0])), skiprows=4)
pop_cols = [str(y) for y in range(2000, 2026)]
population = population.melt(id_vars=['Country Name', 'Country Code'], value_vars=pop_cols,
                             var_name='Year', value_name='Population_WB')
population['Year'] = population['Year'].astype(int)
population['Population_WB'] = pd.to_numeric(population['Population_WB'], errors='coerce')
population = population.rename(columns={'Country Code': 'ISO3', 'Country Name': 'Population_source_country'})
population = population[['ISO3', 'Year', 'Population_WB', 'Population_source_country']]
assert not population.duplicated(['ISO3', 'Year']).any(), 'Duplicate population ISO-year!'

# 4. Countries are identical in WHO measles and vaccination files for the 195
# countries with WUENIC estimates. Join on official WHO country names, then
# population on WUENIC ISO3 codes. For territories lacking WUENIC, population
# is left missing instead of guessing mappings.
panel = cases.merge(vax, on=['Country', 'Year'], how='left', validate='one_to_one')
# Some WHO territories have no WUENIC records but DO have a World Bank
# population series. Map these to WB codes without inventing vaccination data.
wb_country_codes = (population[['Population_source_country', 'ISO3']]
                    .drop_duplicates().set_index('Population_source_country')['ISO3'].to_dict())
territory_aliases = {
    'China, Hong Kong SAR': 'Hong Kong SAR, China',
    'China, Macao SAR': 'Macao SAR, China',
    'Curaçao': 'Curacao',
    'Sint Maarten (Dutch part)': 'Sint Maarten (Dutch part)',
}
no_iso = panel['ISO3'].isna()
lookups = panel.loc[no_iso, 'Country'].replace(territory_aliases)
panel.loc[no_iso, 'ISO3'] = lookups.map(wb_country_codes).values
panel = panel.merge(population, on=['ISO3', 'Year'], how='left', validate='many_to_one')
panel['Incidence_per_million'] = np.where(
    (panel['Population_WB'] > 0) & panel['Measles_reported_cases'].notna(),
    panel['Measles_reported_cases'] / panel['Population_WB'] * 1_000_000,
    np.nan
)
panel = panel.sort_values(['Country', 'Year']).reset_index(drop=True)
panel['Cases_reported_flag'] = panel['Measles_reported_cases'].notna()
panel['Both_vaccines_available_flag'] = panel[['MCV1_WUENIC_pct', 'MCV2_WUENIC_pct']].notna().all(axis=1)
panel['Complete_cases_vacc_population_flag'] = panel[['Measles_reported_cases', 'MCV1_WUENIC_pct',
                                                     'MCV2_WUENIC_pct', 'Population_WB']].notna().all(axis=1)

out_path = OUT / 'Measles_Master_Rebuilt.csv'
panel.to_csv(out_path, index=False)
summary = pd.DataFrame([
    ['Countries/territories', panel['Country'].nunique()],
    ['Country-year rows', len(panel)],
    ['Cases observed', int(panel['Cases_reported_flag'].sum())],
    ['Cases + WUENIC MCV1 + MCV2', int((panel['Cases_reported_flag'] & panel['Both_vaccines_available_flag']).sum())],
    ['Cases + vaccines + WB population', int(panel['Complete_cases_vacc_population_flag'].sum())],
    ['Incidence available', int(panel['Incidence_per_million'].notna().sum())],
    ['Country-year duplicate rows', int(panel.duplicated(['Country', 'Year']).sum())],
], columns=['Metric', 'Value'])
summary.to_csv(RESULTS / 'Step1_Rebuild_Audit.csv', index=False)
print(summary.to_string(index=False))
print(f'\nSaved: {out_path.relative_to(ROOT)}')
print('NOTE: WHO 2026-revised vaccination estimates are retrospective, not historical real-time vintages.')
