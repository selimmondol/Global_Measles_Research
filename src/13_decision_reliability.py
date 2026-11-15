#!/usr/bin/env python3
"""Step 13: validation-only annual decision reliability and protocol freeze.

Run from project root:
    python src/13_decision_reliability.py

Requires Steps 6, 8, 10 validation-prediction outputs. Does not read test data.
Bootstrap samples COUNTRIES with replacement, retaining their available
validation years; each draw ranks sampled country copies within each year.
Only 3 calendar years are observed: these are descriptive cluster bootstrap
intervals, not robust uncertainty for arbitrary future global year shocks.
"""
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'results/baseline_models/validation_predictions.csv'
ADV = ROOT / 'results/advanced_ml_models/validation_predictions.csv'
FEATURES = ROOT / 'data/processed/Measles_Model_Features.csv'
OUT = ROOT / 'results/decision_reliability'
YEARS = (2019, 2020, 2021)
COHORT = 'all_eligible_imputed_vaccination'
MODELS = ('vaccination_only','combined','rf_regularized','xgb_regularized')
APPROACHES = MODELS + ('coverage_shortfall_rank',)
CAPACITIES = (0.10, 0.20)
BOOTSTRAP_DRAWS = 500
SEED = 20261110


def read_predictions(path, models):
    if not path.exists():
        raise SystemExit(f'Missing {path}. Run prior baseline/advanced model script.')
    p = pd.read_csv(path)
    cols = ['cohort','model','Country','Year','observed_surge_label','predicted_probability']
    absent = set(cols) - set(p.columns)
    if absent:
        raise SystemExit(f'{path}: missing columns {sorted(absent)}')
    p = p.loc[(p.cohort == COHORT) & p.model.isin(models), cols].copy()
    if p.empty or set(p.model.unique()) != set(models):
        raise SystemExit(f'{path}: required models absent')
    if not p.Year.isin(YEARS).all():
        raise SystemExit(f'STOP: {path} has non-validation years')
    if p[cols].isna().any().any() or p.duplicated(['model','Country','Year']).any():
        raise SystemExit(f'{path}: nulls or duplicate predictions')
    if not p.observed_surge_label.isin([0,1]).all() or not p.predicted_probability.between(0,1).all():
        raise SystemExit(f'{path}: invalid outcomes or probabilities')
    return p


def load_validation():
    b = read_predictions(BASE, MODELS[:2])
    a = read_predictions(ADV, MODELS[2:])
    p = pd.concat([b,a], ignore_index=True)
    wide = p.pivot(index=['Country','Year','observed_surge_label'],
                   columns='model', values='predicted_probability')
    if len(wide) * len(MODELS) != len(p) or wide.isna().any().any():
        raise SystemExit('STOP: model-country-year predictions not aligned; check labels and row counts')
    wide = wide.reset_index()
    if wide.duplicated(['Country','Year']).any():
        raise SystemExit('STOP: outcome disagreement across models')
    if not FEATURES.exists():
        raise SystemExit(f'Missing {FEATURES}')
    v = pd.read_csv(FEATURES, usecols=['Country','Year','mcv1_lag1','mcv2_lag1'])
    v = v.loc[v.Year.isin(YEARS)].copy()
    if v.duplicated(['Country','Year']).any():
        raise SystemExit('STOP: duplicate feature-country-years')
    wide = wide.merge(v, on=['Country','Year'], how='left', validate='one_to_one', indicator=True)
    if not wide['_merge'].eq('both').all():
        raise SystemExit('STOP: missing feature merge records')
    wide = wide.drop(columns=['_merge'])
    observed = wide[['mcv1_lag1','mcv2_lag1']].notna().all(axis=1)
    wide['high_coverage'] = observed & wide.mcv1_lag1.ge(95) & wide.mcv2_lag1.ge(95)
    wide['coverage_shortfall_rank'] = (100 - wide[['mcv1_lag1','mcv2_lag1']].min(axis=1)).astype(float)
    wide.loc[~observed, 'coverage_shortfall_rank'] = -np.inf
    return wide


def evaluate_years(data, capacity, group_col='Country', include_rows=False):
    """Rank each year independently, using a deterministic sample key tie break."""
    summaries = []
    all_selected = []
    for yr in YEARS:
        d = data.loc[data.Year.eq(yr)]
        if len(d) == 0 or d[group_col].duplicated().any():
            raise ValueError(f'Invalid duplicate or missing sampling units in {yr}')
        k = max(1, math.ceil(len(d)*capacity))
        y = d.observed_surge_label.astype(int)
        for approach in APPROACHES:
            ranked = d.sort_values([approach, group_col], ascending=[False,True],kind='stable')
            chosen = ranked.head(k)
            n_events = int(y.sum())
            tp = int(chosen.observed_surge_label.sum())
            hv = int(d.loc[d.high_coverage, 'observed_surge_label'].sum())
            hv_tp = int(chosen.loc[chosen.high_coverage, 'observed_surge_label'].sum())
            summaries.append(dict(Year=yr, approach=approach, capacity_pct=int(capacity*100),
                                  countries=len(d), selected=k, events=n_events, detected=tp,
                                  missed=n_events-tp, false_alarms=k-tp,
                                  high_coverage_events=hv, high_coverage_detected=hv_tp))
            if include_rows:
                mini = chosen[['Country','Year','observed_surge_label','high_coverage']].copy()
                mini['approach'] = approach
                mini['capacity_pct'] = int(capacity*100)
                all_selected.append(mini)
    return pd.DataFrame(summaries), (pd.concat(all_selected, ignore_index=True)
                                     if include_rows else None)


def aggregate(annual):
    df = annual.groupby(['approach','capacity_pct'], as_index=False)[
        ['countries','selected','events','detected','missed','false_alarms',
         'high_coverage_events','high_coverage_detected']].sum()
    df['recall'] = df.detected / df.events.replace(0,np.nan)
    df['precision'] = df.detected / df.selected.replace(0,np.nan)
    df['false_alarm_share'] = df.false_alarms / df.selected.replace(0,np.nan)
    df['high_coverage_recall'] = df.high_coverage_detected / df.high_coverage_events.replace(0,np.nan)
    return df


def bootstrapped_scores(original):
    """Resample entire countries (including years); duplicated country gets unique draw ID."""
    rng = np.random.default_rng(SEED)
    countries = np.sort(original.Country.unique())
    by_country = {c: original.loc[original.Country.eq(c)] for c in countries}
    draws = []
    for b in range(BOOTSTRAP_DRAWS):
        samples = rng.choice(countries, size=len(countries), replace=True)
        parts = []
        for j, name in enumerate(samples):
            one = by_country[name].copy()
            one['bootstrap_unit'] = f'{j:05d}'  # unique even if same country sampled repeatedly
            parts.append(one)
        sample = pd.concat(parts, ignore_index=True)
        for capacity in CAPACITIES:
            annual, _ = evaluate_years(sample, capacity, group_col='bootstrap_unit')
            pooled = aggregate(annual)
            rec = dict(draw=b, capacity_pct=int(capacity*100))
            for approach in APPROACHES:
                row = pooled.loc[pooled.approach.eq(approach)].iloc[0]
                rec[f'{approach}_recall'] = float(row.recall)
                rec[f'{approach}_precision'] = float(row.precision)
                rec[f'{approach}_high_coverage_recall'] = float(row.high_coverage_recall)
            draws.append(rec)
    return pd.DataFrame(draws)


def main():
    data = load_validation()
    if len(data) != 396 or int(data.observed_surge_label.sum()) != 87:
        raise SystemExit('STOP: Expected 396 validation country-years and 87 events; review inputs')
    annuals, select_rows = [], []
    for c in CAPACITIES:
        annual, selected = evaluate_years(data, c, include_rows=True)
        annuals.append(annual)
        select_rows.append(selected)
    annual = pd.concat(annuals, ignore_index=True)
    pooled = aggregate(annual)
    boot = bootstrapped_scores(data)
    contrasts = []
    comparisons = [('rf_regularized','coverage_shortfall_rank'),
                   ('rf_regularized','vaccination_only'),
                   ('rf_regularized','combined'),
                   ('rf_regularized','xgb_regularized'),
                   ('xgb_regularized','coverage_shortfall_rank')]
    for capacity in (10,20):
        sub = boot.loc[boot.capacity_pct.eq(capacity)]
        observed = pooled.loc[pooled.capacity_pct.eq(capacity)].set_index('approach')
        for a,b in comparisons:
            for metric in ('recall','precision','high_coverage_recall'):
                diff = sub[f'{a}_{metric}'] - sub[f'{b}_{metric}']
                diff = diff.replace([np.inf,-np.inf],np.nan).dropna()
                point = float(observed.loc[a,metric]-observed.loc[b,metric])
                contrasts.append(dict(capacity_pct=capacity, model_a=a, model_b=b,
                                      metric=metric, observed_difference=point,
                                      lower_95pct=float(diff.quantile(.025)) if len(diff) else np.nan,
                                      upper_95pct=float(diff.quantile(.975)) if len(diff) else np.nan,
                                      valid_draws=len(diff)))
    OUT.mkdir(parents=True, exist_ok=True)
    annual.to_csv(OUT/'annual_decision_metrics_checked.csv', index=False)
    pooled.to_csv(OUT/'pooled_decision_summary_checked.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'paired_decision_bootstrap.csv', index=False)
    pd.concat(select_rows,ignore_index=True).to_csv(OUT/'annual_selected_country_years.csv', index=False)
    # Pre-specified choice using validation only: choose RF for primary operational
    # comparison (highest observed annual recall@10% among ML contenders).
    # This is a PROTOCOL proposal; freeze it before any later test access.
    protocol = dict(
        script='13_decision_reliability.py', prediction_years=list(YEARS),
        cohort=COHORT, primary_operating_capacity_pct=10, secondary_capacity_pct=20,
        primary_model='rf_regularized', secondary_model='xgb_regularized',
        statistical_baseline='combined', conventional_comparator='coverage_shortfall_rank',
        primary_metric='pooled annual recall at 10% per-year selected countries',
        selection_rule='Per year ceil(10% of eligible countries), descending score; country ascending on ties',
        endpoint='Provisional reported annual measles incidence surge Definition C; not WHO outbreak',
        bootstrap='500 country-cluster draws; calendar shocks not independently captured',
        test_rule='Do not access 2022-2025 future or unseen-country test until protocol frozen',
        caution='WUENIC revised current-vintage estimates are retrospective; no real-time performance claim')
    (OUT/'protocol_lock_draft.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    (OUT/'decision_reliability_notes.txt').write_text(
        'VALIDATION ONLY (2019-2021); the test partitions are not read.\n'
        'Bootstrapping re-ranks sampled country copies within each year.\n'
        'Intervals are descriptive and do not account for common global year shocks with only three years.\n'
        'Recall@10% is annual country-year event detection, not a proof of intervention effect.\n'
        'Vaccination shortfall is a simple benchmark, not the WHO official risk tool.\n'
        'Vaccination missingness is ranked last in the conventional benchmark.\n'
        'High coverage (both WUENIC doses >=95%) is a national estimate, not immunity.\n'
        'Prior to test: freeze this protocol AND operational target definition; avoid retrospective tuning.\n',
        encoding='utf-8')
    print('\nSTEP 13: ANNUAL DECISION RELIABILITY (VALIDATION ONLY)\n')
    print(pooled.loc[pooled.capacity_pct.eq(10),
          ['approach','selected','events','detected','false_alarms','recall','precision']].round(4).to_string(index=False))
    print('\nCountry-cluster paired bootstrap: Recall difference, 10% capacity')
    contrast = pd.DataFrame(contrasts)
    print(contrast.loc[(contrast.capacity_pct.eq(10)) & contrast.metric.eq('recall'),
          ['model_a','model_b','observed_difference','lower_95pct','upper_95pct']].round(4).to_string(index=False))
    print(f'\nReports saved: {OUT.relative_to(ROOT)}/')
    print('NOTE: Future-year and unseen-country tests were NOT accessed.')

if __name__=='__main__':
    main()
