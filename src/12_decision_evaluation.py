#!/usr/bin/env python3
"""Step 12 — validation-only annual country-prioritization evaluation.

Run from the Measles project root:
    python src/12_decision_evaluation.py

Inputs (created by Steps 6, 8 and 10):
    results/baseline_models/validation_predictions.csv
    results/advanced_ml_models/validation_predictions.csv
    data/processed/Measles_Model_Features.csv

Outputs: results/decision_evaluation/*.csv and decision_notes.txt

IMPORTANT: Reads ONLY 2019–2021 predictions and vaccination information. Does
not train models, touch the 2022–2025 test outcomes, or change datasets.
Labels are researcher-defined reported-incidence surges, not WHO outbreaks.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'results/baseline_models/validation_predictions.csv'
ADV = ROOT / 'results/advanced_ml_models/validation_predictions.csv'
FEATURE = ROOT / 'data/processed/Measles_Model_Features.csv'
OUT = ROOT / 'results/decision_evaluation'
COHORTS = ('matched_complete_vaccination', 'all_eligible_imputed_vaccination')
MODELS = ('vaccination_only', 'incidence_history', 'combined',
          'rf_regularized', 'xgb_regularized', 'rf_shallow', 'xgb_shallow')
CAPACITIES = (0.10, 0.20)
YEARS = (2019, 2020, 2021)
KEY = ['cohort', 'Country', 'Year']


def load_predictions(path, wanted):
    if not path.is_file():
        raise SystemExit(f'Missing {path}. Run Steps 8 and 10 first.')
    data = pd.read_csv(path, low_memory=False)
    required = {'cohort', 'model', 'Country', 'Year',
                'observed_surge_label', 'predicted_probability'}
    if required - set(data.columns):
        raise SystemExit(f'{path.name}: missing {sorted(required - set(data.columns))}')
    data = data.loc[data['model'].isin(wanted),
                    ['cohort', 'model', 'Country', 'Year',
                     'observed_surge_label', 'predicted_probability']].copy()
    if data.empty:
        raise SystemExit(f'No expected models in {path.name}')
    if data[['cohort', 'model', 'Country', 'Year', 'observed_surge_label',
             'predicted_probability']].isna().any().any():
        raise SystemExit(f'{path.name}: null keys, outcome or probability')
    if not data.Year.isin(YEARS).all():
        raise SystemExit(f'STOP: {path.name} includes non-validation years')
    if not data.cohort.isin(COHORTS).all():
        raise SystemExit(f'{path.name}: unexpected cohort')
    if data.duplicated(['cohort', 'model', 'Country', 'Year']).any():
        raise SystemExit(f'{path.name}: duplicate model-country-year rows')
    if not data.observed_surge_label.isin([0, 1]).all():
        raise SystemExit(f'{path.name}: labels not binary')
    if not data.predicted_probability.between(0, 1).all():
        raise SystemExit(f'{path.name}: probabilities outside [0, 1]')
    return data


def top_k_counts(one, capacity, score_col='predicted_probability'):
    """Rank unique countries within a single year. Deterministic tie-break."""
    if one.Country.duplicated().any() or one.Year.nunique() != 1:
        raise ValueError('Expected exactly one validation observation per country/year')
    n = len(one)
    k = max(1, int(math.ceil(capacity * n)))
    ranked = one.sort_values([score_col, 'Country'], ascending=[False, True],
                             kind='stable').copy()
    ranked['rank_within_year'] = np.arange(1, n + 1)
    ranked['selected'] = ranked.rank_within_year.le(k)
    selected = ranked.loc[ranked.selected]
    events = int(ranked.observed_surge_label.sum())
    tp = int(selected.observed_surge_label.sum())
    fp = int(k - tp)
    return ranked, dict(n=n, k=k, events=events, detected=tp,
                        missed=events-tp, false_alarms=fp,
                        precision=tp/k, recall=tp/events if events else np.nan)


def summarise(decisions):
    by = ['cohort', 'approach', 'capacity_pct']
    totals = decisions.groupby(by, dropna=False, as_index=False)[
        ['countries_evaluated', 'selected', 'events', 'detected', 'missed',
         'false_alarms', 'high_coverage_events', 'high_coverage_detected']
    ].sum()
    totals['recall'] = totals.detected / totals.events.replace(0, np.nan)
    totals['precision'] = totals.detected / totals.selected.replace(0, np.nan)
    totals['false_alarm_share'] = totals.false_alarms / totals.selected.replace(0, np.nan)
    totals['high_coverage_recall'] = (totals.high_coverage_detected /
                                      totals.high_coverage_events.replace(0, np.nan))
    totals['num_years'] = len(YEARS)
    return totals


def main():
    baseline = load_predictions(BASE, MODELS[:3])
    advanced = load_predictions(ADV, MODELS[3:])
    d = pd.concat([baseline, advanced], ignore_index=True)
    if d.duplicated(['cohort', 'model', 'Country', 'Year']).any():
        raise SystemExit('STOP: duplicated predictions from combined sources')

    # Read vaccination for validation YEARS ONLY. Never read test outcomes or
    # holdout assignments. Prediction files are already quarantined to validation.
    if not FEATURE.is_file():
        raise SystemExit(f'Missing {FEATURE}. Run Step 6 first.')
    vaccine = pd.read_csv(FEATURE, usecols=['Country', 'Year', 'mcv1_lag1', 'mcv2_lag1'])
    vaccine = vaccine.loc[vaccine.Year.isin(YEARS)].copy()
    if vaccine.duplicated(['Country', 'Year']).any():
        raise SystemExit('STOP: feature table contains duplicate country-years')
    d = d.merge(vaccine, on=['Country', 'Year'], validate='many_to_one',
                how='left', indicator=True)
    if not d['_merge'].eq('both').all():
        raise SystemExit('STOP: incomplete vaccination-feature alignment')
    d = d.drop(columns='_merge')
    d['both_vax_observed'] = d[['mcv1_lag1', 'mcv2_lag1']].notna().all(axis=1)
    d['high_vax'] = (d.both_vax_observed & d.mcv1_lag1.ge(95) & d.mcv2_lag1.ge(95))
    d['coverage_below_95'] = (d.both_vax_observed &
                               (d.mcv1_lag1.lt(95) | d.mcv2_lag1.lt(95)))
    # Larger shortfall means greater vaccination coverage concern. NaN is
    # UNRANKABLE; when equal capacity is required place missing at the bottom.
    d['coverage_shortfall_score'] = (100.0 - d[['mcv1_lag1', 'mcv2_lag1']].min(axis=1))
    d.loc[~d.both_vax_observed, 'coverage_shortfall_score'] = -np.inf

    alignment, annual, selections, coverage_threshold = [], [], [], []
    for cohort in COHORTS:
        by_model = {}
        for model in MODELS:
            one = d.loc[d.cohort.eq(cohort) & d.model.eq(model)].sort_values(
                ['Year', 'Country']).reset_index(drop=True)
            if one.empty:
                raise SystemExit(f'Missing predictions for {cohort}/{model}')
            by_model[model] = one
        reference = by_model['vaccination_only']
        for model, one in by_model.items():
            if not reference[['Country', 'Year', 'observed_surge_label']].equals(
                one[['Country', 'Year', 'observed_surge_label']]):
                raise SystemExit(f'STOP: country-year or label mismatch: {cohort}/{model}')
        if cohort == 'matched_complete_vaccination' and not reference.both_vax_observed.all():
            raise SystemExit('STOP: complete-vaccine cohort contains missing vaccine values')
        alignment.append(dict(cohort=cohort, n=len(reference),
                              events=int(reference.observed_surge_label.sum()),
                              years=';'.join(str(x) for x in YEARS),
                              all_models_identical_country_year_labels=True))

        for yr in YEARS:
            ref_year = reference.loc[reference.Year.eq(yr)].copy()
            if not len(ref_year) or ref_year.Country.duplicated().any():
                raise SystemExit(f'Invalid country rows in {cohort}/{yr}')
            # Fixed 95%-rule, not capacity constrained; missing vaccination is
            # UNKNOWN rather than assumed low/high coverage.
            known = ref_year.loc[ref_year.both_vax_observed]
            flagged = known.loc[known.coverage_below_95]
            coverage_threshold.append(dict(
                cohort=cohort, Year=yr, n_all=len(ref_year),
                n_coverage_known=len(known), n_coverage_unknown=len(ref_year)-len(known),
                observed_events_all=int(ref_year.observed_surge_label.sum()),
                observed_events_coverage_known=int(known.observed_surge_label.sum()),
                flagged_below_95=len(flagged),
                detected_below_95=int(flagged.observed_surge_label.sum()),
                false_alarms_below_95=int(len(flagged)-flagged.observed_surge_label.sum()),
                missed_with_known_coverage=int(known.observed_surge_label.sum()-flagged.observed_surge_label.sum()),
                positive_with_missing_coverage=int(ref_year.loc[~ref_year.both_vax_observed,
                                                              'observed_surge_label'].sum())))

            # Equal-capacity conventional coverage-shortfall ranking:
            # not an official WHO risk tool and not an epidemiological score.
            strategies = {m: (by_model[m].loc[by_model[m].Year.eq(yr)].copy(),
                              'predicted_probability') for m in MODELS}
            strategies['coverage_shortfall_rank'] = (ref_year.copy(), 'coverage_shortfall_score')
            for approach, (one_year, score_col) in strategies.items():
                if not one_year.Country.sort_values().reset_index(drop=True).equals(
                    ref_year.Country.sort_values().reset_index(drop=True)):
                    raise SystemExit(f'Missing strategy countries: {cohort}/{yr}/{approach}')
                for capacity in CAPACITIES:
                    ranked, score = top_k_counts(one_year, capacity, score_col)
                    hv = ranked.high_vax
                    score['high_coverage_events'] = int(ranked.loc[hv,'observed_surge_label'].sum())
                    score['high_coverage_detected'] = int(ranked.loc[hv & ranked.selected,
                                                                    'observed_surge_label'].sum())
                    annual.append(dict(cohort=cohort, Year=yr, approach=approach,
                                       capacity_pct=int(capacity*100),
                                       countries_evaluated=score.pop('n'),
                                       selected=score.pop('k'), **score))
                    # Save only selected rows, plus unselected TRUE events for
                    # transparent missed-event audit. All are validation years.
                    keep = ranked.selected | ranked.observed_surge_label.eq(1)
                    ranked = ranked.loc[keep].copy()
                    ranked['cohort'] = cohort
                    ranked['approach'] = approach
                    ranked['capacity_pct'] = int(capacity*100)
                    selections.append(ranked[['cohort','approach','capacity_pct',
                                              'Country','Year','observed_surge_label',
                                              'rank_within_year','selected',score_col,
                                              'both_vax_observed','high_vax']].rename(
                                                  columns={score_col:'ranking_score'}))

    annual_df = pd.DataFrame(annual)
    pooled_df = summarise(annual_df)
    threshold_df = pd.DataFrame(coverage_threshold)
    # For a fixed coverage threshold, counts vary by annual coverage prevalence;
    # don't compare directly to fixed-capacity ML without reporting selection size.
    threshold_df['recall_known_coverage'] = (threshold_df.detected_below_95 /
                                             threshold_df.observed_events_coverage_known.replace(0,np.nan))
    threshold_df['precision_below_95'] = (threshold_df.detected_below_95 /
                                          threshold_df.flagged_below_95.replace(0,np.nan))
    selected_df = pd.concat(selections, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    annual_df.to_csv(OUT/'annual_decision_metrics.csv', index=False)
    pooled_df.to_csv(OUT/'pooled_decision_summary.csv', index=False)
    threshold_df.to_csv(OUT/'coverage_threshold_screen.csv', index=False)
    selected_df.to_csv(OUT/'selected_and_missed_country_years.csv', index=False)
    pd.DataFrame(alignment).to_csv(OUT/'alignment_checks.csv', index=False)
    (OUT/'decision_notes.txt').write_text(
        'VALIDATION ONLY: 2019–2021. Test and unseen-country partitions not accessed.\n'
        'Annual top-K selects CEIL(10% or 20% of eligible countries) separately each year.\n'
        'Pooled recall = sum detected / sum events across years; not average annual recall.\n'
        'Tie breaking: descending score, ascending country name.\n'
        'Coverage-shortfall ranking uses 100 - min(previous-year MCV1, MCV2);\n'
        'missing coverage is ranked last, not assigned coverage=0. This is a simple\n'
        'research comparator, NOT the WHO risk-assessment framework.\n'
        'Fixed <95% coverage screen uses only countries with both vaccines observed;\n'
        'no imputation of vaccine coverage for this rule. Capacity is not fixed.\n'
        'Per-year target is a researcher-defined annual incidence surge, NOT\n'
        'WHO-defined outbreak onset. Reported cases are surveillance-dependent.\n'
        'All 2019–2021 comparisons are validation/exploratory, used repeatedly in\n'
        'development. Do not claim unbiased out-of-sample improvement or significance.\n'
        'Present-day WUENIC revisions do not imply historical real-time availability.\n'
        'Country-year observations across calendar years are dependent.\n'
        'Selection by validation performance must be frozen before 2022–2025 test.\n',
        encoding='utf-8')
    print('\nSTEP 12: ANNUAL COUNTRY PRIORITIZATION (VALIDATION ONLY)\n')
    preview = pooled_df.loc[(pooled_df.cohort=='all_eligible_imputed_vaccination') &
                            (pooled_df.capacity_pct==10),
                            ['approach','countries_evaluated','selected','events',
                             'detected','missed','false_alarms','recall','precision']]
    print(preview.round(4).to_string(index=False))
    print('\nFixed 95% coverage screen (not equal capacity):')
    print(threshold_df.loc[threshold_df.cohort=='all_eligible_imputed_vaccination',
                            ['Year','n_coverage_known','n_coverage_unknown',
                             'flagged_below_95','detected_below_95']].to_string(index=False))
    print(f'\nReports saved: {OUT.relative_to(ROOT)}/')
    print('NOTE: No models trained. All outcomes & predictors are from validation 2019–2021 only.')


if __name__ == '__main__':
    main()
