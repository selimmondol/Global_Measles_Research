#!/usr/bin/env python3
"""Step 4: read-only validation of provisional measles resurgence labels.

Run from the Measles project root:
    python src/04_outcome_validation.py

Input: data/processed/Measles_Resurgence_Candidates.csv
Output: results/outcome_validation/*.csv and validation_notes.txt

No model training, no target modification, no imputation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data' / 'processed' / 'Measles_Resurgence_Candidates.csv'
OUT = ROOT / 'results' / 'outcome_validation'
LABELS = ['surge_A_2x_1pm', 'surge_B_3x_1pm', 'surge_C_2x_5pm', 'surge_D_2x_5pm_2increase']


def save(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


def main():
    if not SOURCE.is_file():
        raise SystemExit(f'Missing input: {SOURCE}\nRun 03_resurgence_definition.py first.')
    df = pd.read_csv(SOURCE)
    needed = ['Country', 'Year', 'Incidence_per_million', 'history_3y_complete',
              'prior_3y_median_per_million', 'outcome_eligible',
              'prior_both_mcv_ge95', *LABELS]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise SystemExit(f'Missing expected columns: {absent}')
    OUT.mkdir(parents=True, exist_ok=True)
    if df.duplicated(['Country','Year']).any():
        raise SystemExit('Duplicate Country-Year rows: audit input before proceeding.')

    df['Year'] = pd.to_numeric(df['Year'], errors='raise').astype(int)
    df = df.sort_values(['Country', 'Year']).copy()
    incidence_missing = df['Incidence_per_million'].isna()
    history_missing = ~df['history_3y_complete'].fillna(False).astype(bool)
    eligible = df['outcome_eligible'].fillna(False).astype(bool)
    expected = ~incidence_missing & ~history_missing
    if not eligible.equals(expected):
        raise SystemExit('Outcome eligibility disagrees with missing current incidence/history.')
    for c in LABELS:
        if df.loc[eligible, c].isna().any() or df.loc[~eligible, c].notna().any():
            raise SystemExit(f'{c}: labels must be populated only for eligible rows.')
        if not df.loc[eligible, c].isin([0, 1]).all():
            raise SystemExit(f'{c}: unexpected label values.')

    exclusive = [
        ('Missing current incidence AND prior history', incidence_missing & history_missing),
        ('Missing current incidence ONLY', incidence_missing & ~history_missing),
        ('Incomplete prior 3-year history ONLY', ~incidence_missing & history_missing),
        ('Eligible: current incidence and 3-year history available', expected),
    ]
    save([{'Category': title, 'Rows': int(mask.sum())} for title, mask in exclusive],
         'eligibility_exclusive.csv')

    median = df['prior_3y_median_per_million']
    zero_median = eligible & median.eq(0)
    save([{'Group':'Eligible total', 'Rows':int(eligible.sum())},
          {'Group':'Eligible with zero historical median','Rows':int(zero_median.sum())},
          {'Group':'Eligible with positive historical median','Rows':int((eligible & median.gt(0)).sum())},
          *[{'Group':f'{c}: events at zero baseline',
             'Rows':int((zero_median & df[c].eq(1)).sum())} for c in LABELS]],
         'zero_baseline_summary.csv')

    # Pairwise comparison: same event count does not necessarily mean identical labels.
    pairs = []
    for i, a in enumerate(LABELS):
        for b in LABELS[i+1:]:
            a1 = df.loc[eligible, a].eq(1)
            b1 = df.loc[eligible, b].eq(1)
            both = int((a1 & b1).sum())
            union = int((a1 | b1).sum())
            pairs.append({'Definition_1':a,'Definition_2':b,'N_eligible':int(eligible.sum()),
                          'Both_event':both,'Only_definition_1':int((a1 & ~b1).sum()),
                          'Only_definition_2':int((~a1 & b1).sum()),
                          'Neither_event':int((~a1 & ~b1).sum()),
                          'Disagreements':int((a1 != b1).sum()),
                          'Jaccard_event_agreement':round(both/union, 5) if union else np.nan})
    save(pairs, 'definition_pairwise_agreement.csv')

    # Consecutive event years are not independent outbreaks. We report episodes only
    # as provisional runs of adjacent labelled years, NOT certified outbreak counts.
    episodes = []
    country_rows = []
    for country, group in df.groupby('Country', sort=True):
        group = group.sort_values('Year')
        for label in LABELS[:3]:
            ev = group.loc[group['outcome_eligible'] & group[label].eq(1), 'Year'].tolist()
            consecutive_pairs = sum(b == a + 1 for a, b in zip(ev, ev[1:]))
            country_rows.append({'Country':country,'Definition':label,
                                 'Event_years':len(ev),'Adjacent_event_year_pairs':consecutive_pairs})
            if ev:
                start = prev = ev[0]
                for year in ev[1:]:
                    if year != prev+1:
                        episodes.append({'Country':country,'Definition':label,
                                         'Start_year':start,'End_year':prev,
                                         'Duration_years':prev-start+1})
                        start=year
                    prev=year
                episodes.append({'Country':country,'Definition':label,
                                 'Start_year':start,'End_year':prev,
                                 'Duration_years':prev-start+1})
    save(country_rows,'events_by_country_validated.csv')
    save(episodes,'consecutive_event_runs.csv')

    # Distribution over time and in pre-existing high-vaccination strata.
    yearly = []
    for year, g in df.groupby('Year',sort=True):
        ok=g['outcome_eligible']
        high=g['prior_both_mcv_ge95'].eq(True) & ok
        item={'Year':int(year),'Eligible':int(ok.sum()),'High_MCV_eligible':int(high.sum())}
        for c in LABELS[:3]:
            item[c+'_events']=int((ok & g[c].eq(1)).sum())
            item[c+'_high_MCV_events']=int((high & g[c].eq(1)).sum())
        yearly.append(item)
    save(yearly,'events_by_year_validated.csv')

    compare_cd = next(x for x in pairs if x['Definition_1']==LABELS[2] and x['Definition_2']==LABELS[3])
    n_episodes_c=sum(e['Definition']==LABELS[2] for e in episodes)
    summary=[
        ('Input country-year rows',len(df)),('Outcome eligible',int(eligible.sum())),
        ('Unique ineligible',int((~eligible).sum())),
        ('Eligible zero-median rows',int(zero_median.sum())),
        ('Definition A events',int((eligible&df[LABELS[0]].eq(1)).sum())),
        ('Definition B events',int((eligible&df[LABELS[1]].eq(1)).sum())),
        ('Definition C events',int((eligible&df[LABELS[2]].eq(1)).sum())),
        ('Definition D events',int((eligible&df[LABELS[3]].eq(1)).sum())),
        ('C vs D disagreement rows',compare_cd['Disagreements']),
        ('Definition C provisional consecutive-year runs',n_episodes_c),
    ]
    save([{'Metric':k,'Value':v} for k,v in summary], 'outcome_validation_summary.csv')
    (OUT/'validation_notes.txt').write_text(
        'STEP 4 / OUTCOME VALIDATION\n'
        'These are researcher-defined reported-incidence surge labels, NOT WHO-confirmed outbreaks.\n'
        'All categories in eligibility_exclusive.csv are mutually exclusive.\n'
        'Cases with a zero historical median need special care; ratio-based rules can be misleading.\n'
        'Consecutive-year runs summarize adjacent positive labels; they are NOT validated outbreak episodes.\n'
        'Vaccination >=95% is NATIONAL prior-year WUENIC coverage, not proof of individual immunity.\n'
        'Study limitations include reporting heterogeneity, current-revision WUENIC estimates,\n'
        'missing population for some territories and repeated within-country observations.\n'
        'No definition is established here as the final primary endpoint.\n',encoding='utf-8')

    print('\nSTEP 4: OUTCOME VALIDATION (read only)\n')
    for k,v in summary: print(f'{k:<52} {v:>7}')
    print(f'\nReports saved: {OUT.relative_to(ROOT)}/')
    print('NOTE: Labels are provisional. No original files or targets were overwritten.')

if __name__ == '__main__':
    main()
