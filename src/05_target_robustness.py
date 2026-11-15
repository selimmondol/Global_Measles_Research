#!/usr/bin/env python3
"""Step 5: read-only robustness audit of provisional annual measles incidence surges.

Run from the Measles project root:
    python src/05_target_robustness.py

Input: data/processed/Measles_Resurgence_Candidates.csv
Output: results/target_robustness/*.csv and interpretation_notes.txt

This does not modify existing data or claim to identify WHO-confirmed outbreaks.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data' / 'processed' / 'Measles_Resurgence_Candidates.csv'
REPORTS = ROOT / 'results' / 'target_robustness'
BASE = 'surge_C_2x_5pm'
REQUIRED = ['Country', 'Year', 'Incidence_per_million',
            'prior_3y_median_per_million', 'outcome_eligible',
            'prior_both_mcv_ge95', BASE]


def save(rows, filename):
    pd.DataFrame(rows).to_csv(REPORTS / filename, index=False)


def main():
    if not SOURCE.is_file():
        raise SystemExit(f'Missing {SOURCE}. Run 03_resurgence_definition.py first.')
    df = pd.read_csv(SOURCE, low_memory=False)
    absent = [col for col in REQUIRED if col not in df.columns]
    if absent:
        raise SystemExit(f'Missing columns: {absent}')
    if df[ ['Country', 'Year'] ].isna().any().any() or df.duplicated(['Country','Year']).any():
        raise SystemExit('Invalid country-year keys. Fix upstream.')
    df['Year'] = pd.to_numeric(df['Year'], errors='raise').astype(int)
    for col in ['Incidence_per_million', 'prior_3y_median_per_million', BASE]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values(['Country', 'Year']).reset_index(drop=True)
    eligible = df['outcome_eligible'].fillna(False).astype(bool)
    rate = df['Incidence_per_million']
    median = df['prior_3y_median_per_million']
    if df.loc[eligible, [*['Incidence_per_million','prior_3y_median_per_million'], BASE]].isna().any().any():
        raise SystemExit('An eligible row has an absent incidence, baseline or C label.')
    if df.loc[~eligible, BASE].notna().any() or not df.loc[eligible, BASE].isin([0,1]).all():
        raise SystemExit('Primary candidate C labels inconsistent with eligibility.')

    # Strictly reproduce C rather than silently changing the endpoint.
    recalc = (rate >= 5) & (rate >= 2*median) & ((rate-median) >= 0.5)
    mismatch = int((recalc.loc[eligible] != df.loc[eligible, BASE].eq(1)).sum())
    if mismatch:
        raise SystemExit(f'{mismatch} C labels disagree with the documented definition.')

    REPORTS.mkdir(parents=True, exist_ok=True)
    pos = eligible & df[BASE].eq(1)
    baseline_zero = eligible & median.eq(0)
    baseline_positive = eligible & median.gt(0)
    zero_pos = pos & baseline_zero

    # Sensitivity: excluding zero-baseline ROWS, not recoding them as negatives.
    save([
        {'Subgroup':'All eligible', 'Eligible':int(eligible.sum()), 'Events_C':int(pos.sum()),
         'Event_rate_pct':round(100*pos.sum()/eligible.sum(), 3)},
        {'Subgroup':'Zero 3-year median', 'Eligible':int(baseline_zero.sum()),
         'Events_C':int(zero_pos.sum()),
         'Event_rate_pct':round(100*zero_pos.sum()/baseline_zero.sum(), 3)},
        {'Subgroup':'Positive 3-year median only', 'Eligible':int(baseline_positive.sum()),
         'Events_C':int((pos & baseline_positive).sum()),
         'Event_rate_pct':round(100*(pos & baseline_positive).sum()/baseline_positive.sum(), 3)},
    ], 'zero_baseline_robustness.csv')

    # A run start does not necessarily mean that the previous year was observed.
    # Strict onset requires preceding calendar year to be eligible and non-event.
    g = df.groupby('Country', sort=False)
    prev_y = g['Year'].shift(1)
    prev_elig = g['outcome_eligible'].shift(1).eq(True)
    prev_pos = g[BASE].shift(1).eq(1)
    contiguous = (df['Year']-prev_y).eq(1)
    confirmed_prior_negative = contiguous & prev_elig & ~prev_pos
    strict_onset = pos & confirmed_prior_negative
    continuing_positive = pos & contiguous & prev_elig & prev_pos
    unknown_prior = pos & ~(confirmed_prior_negative | (contiguous & prev_elig & prev_pos))
    run_start = pos & ~continuing_positive  # descriptive only, not a validated onset
    assert int(pos.sum()) == int(strict_onset.sum()+continuing_positive.sum()+unknown_prior.sum())
    save([
        {'Category':'Positive C-labelled years', 'Rows':int(pos.sum())},
        {'Category':'Strict new onset: preceding year observed negative', 'Rows':int(strict_onset.sum())},
        {'Category':'Continuation: preceding year observed positive', 'Rows':int(continuing_positive.sum())},
        {'Category':'Previous-year label missing or not eligible', 'Rows':int(unknown_prior.sum())},
        {'Category':'Descriptive run starts (includes unknown starts)', 'Rows':int(run_start.sum())},
    ], 'onset_audit.csv')

    # Fixed, predeclared grid. No parameter chosen from model performance.
    grid=[]
    for threshold in (1., 2., 5., 10.):
        for multiple in (2., 3.):
            test = eligible & rate.ge(threshold) & rate.ge(multiple*median) & (rate-median).ge(0.5)
            nozero = test & baseline_positive
            high = eligible & df['prior_both_mcv_ge95'].eq(1)
            grid.append({'Minimum_incidence_per_million':threshold,
                         'Baseline_multiple':multiple, 'Minimum_increase_per_million':0.5,
                         'Eligible':int(eligible.sum()), 'Events':int(test.sum()),
                         'Event_rate_pct':round(100*test.sum()/eligible.sum(),3),
                         'Events_positive_baseline':int(nozero.sum()),
                         'High_MCV_eligible':int(high.sum()),
                         'High_MCV_events':int((test & high).sum())})
    save(grid, 'threshold_grid.csv')

    # Distribution and denominators, including strict onset counts by calendar year.
    yearly=[]
    for yr, ix in df.groupby('Year').groups.items():
        mask = df.index.isin(ix)
        year_eligible = eligible & mask
        if not year_eligible.any():
            continue
        yearly.append({'Year':int(yr), 'Eligible':int(year_eligible.sum()),
                       'C_events':int((pos & mask).sum()),
                       'C_events_zero_median':int((zero_pos & mask).sum()),
                       'Strict_onsets':int((strict_onset & mask).sum()),
                       'Unknown_previous_label':int((unknown_prior & mask).sum())})
    save(yearly, 'yearly_robustness.csv')

    country=[]
    for name, ix in df.groupby('Country').groups.items():
        mask = df.index.isin(ix)
        country.append({'Country':name,'Eligible':int((eligible & mask).sum()),
                        'C_events':int((pos & mask).sum()),
                        'C_events_zero_median':int((zero_pos & mask).sum()),
                        'Strict_onsets':int((strict_onset & mask).sum())})
    save(country, 'country_robustness.csv')

    (REPORTS/'interpretation_notes.txt').write_text(
        'TARGET ROBUSTNESS — PROVISIONAL REPORTED INCIDENCE SURGES ONLY\n'
        'Primary candidate C: incidence >= 5 per million, >= 2x prior three-year median,\n'
        'and increase >= 0.5 per million. This is a researcher-defined threshold,\n'
        'NOT a WHO outbreak definition and NOT yet a final prespecified endpoint.\n'
        'Zero-baseline sensitivity excludes zero-median rows; it does not recode them as 0.\n'
        'Strict onset needs a verified previous-year negative label; an unknown previous\n'
        'year is neither assumed negative nor counted as a strict onset.\n'
        'Consecutive positive labels are not independent outbreaks.\n'
        'Threshold grid is descriptive and not intended for cherry-picking results.\n'
        'High MCV: national prior-year WUENIC >=95% for both doses; not true immunity.\n'
        'Reported incidence depends on surveillance completeness. Revised historical\n'
        'WUENIC estimates do not establish what was available in each historical year.\n'
        'No models trained; original files and labels unchanged.\n', encoding='utf-8')
    print('\nSTEP 5: TARGET ROBUSTNESS (read-only)\n')
    for label, val in [
        ('Country-year rows',len(df)),('Outcome eligible',int(eligible.sum())),
        ('Candidate C positive years',int(pos.sum())),
        ('C events at zero historical median',int(zero_pos.sum())),
        ('C events with positive historical median',int((pos & baseline_positive).sum())),
        ('Strict new onsets after observed negative year',int(strict_onset.sum())),
        ('Continuations after observed positive year',int(continuing_positive.sum())),
        ('Positive years with unknown preceding-year label',int(unknown_prior.sum())),
        ('Descriptive run starts',int(run_start.sum()))]:
        print(f'{label:<55} {val:>6}')
    print(f'\nReports: {REPORTS.relative_to(ROOT)}/')
    print('NOTE: No final outcome selected. No raw/processed data modified.')

if __name__ == '__main__':
    main()
