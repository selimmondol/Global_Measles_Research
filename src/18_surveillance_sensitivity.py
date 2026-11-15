#!/usr/bin/env python3
"""Step 18: POST-TEST exploratory surveillance/missingness sensitivity.

Read-only. Uses frozen Step 15 predictions, never retrains or tunes models.
This script evaluates PROXIES for reporting completeness, not observed
surveillance quality. It cannot measure WHO/WUENIC data revisions or true
prospective availability from a single current-vintage export.

Run from project root: python src/18_surveillance_sensitivity.py
"""
from pathlib import Path
import math
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / 'results/independent_test_v1/test_predictions.csv'
CANDIDATES = ROOT / 'data/processed/Measles_Resurgence_Candidates.csv'
OUT = ROOT / 'results/surveillance_sensitivity_posttest_v1'
MODELS = ['coverage_shortfall_rank','vaccination_only','combined','rf_regularized','xgb_regularized']


def mandatory(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}\nCheck the path; do not replace it with validation/test outputs from another run.')
    return pd.read_csv(path)


def summary(df, grouping):
    res = []
    for keys, sub in df.groupby(grouping, dropna=False, sort=True):
        if not isinstance(keys, tuple): keys=(keys,)
        record=dict(zip(grouping, keys))
        record.update(n=len(sub), countries=sub.Country.nunique(), events=int(sub.y.sum()),
                      event_rate=float(sub.y.mean()),
                      rf_brier=float(np.mean((sub.rf_regularized-sub.y)**2)),
                      xgb_brier=float(np.mean((sub.xgb_regularized-sub.y)**2)),
                      rf_mean_predicted=float(sub.rf_regularized.mean()),
                      rf_observed_minus_predicted=float(sub.y.mean()-sub.rf_regularized.mean()))
        res.append(record)
    return pd.DataFrame(res)


def select_annual(df, model, pct=10):
    # Frozen evaluation cohort: rank ALL outcome-evaluable country-years each year.
    parts=[]
    for year, sub in df.groupby('Year', sort=True):
        k=int(math.ceil(len(sub)*pct/100))
        ranked=sub.sort_values([model,'Country'], ascending=[False,True], kind='mergesort')
        parts.append(ranked.head(k)[['partition','Country','Year']].assign(approach=model,selected=True))
    return pd.concat(parts, ignore_index=True)


def main():
    p = mandatory(PRED)
    c = mandatory(CANDIDATES)
    required_p={'Country','Year','partition','y',*MODELS,'mcv1_lag1','mcv2_lag1'}
    required_c={'Country','Year','Measles_reported_cases','Incidence_per_million',
                'prior_3y_median_per_million','outcome_eligible','surge_C_2x_5pm'}
    for label, df, needed in [('predictions',p,required_p),('candidate data',c,required_c)]:
        missing=needed-set(df.columns)
        if missing: raise ValueError(f'{label} missing columns: {sorted(missing)}')
        if df[['Country','Year']].duplicated().any(): raise ValueError(f'Duplicate country-year rows in {label}')
    if set(p.partition.unique()) != {'test','unseen_country_test'}:
        raise ValueError('Only locked future and unseen-country test partitions are allowed.')
    if not p.Year.between(2022,2025).all(): raise ValueError('Unexpected prediction years; refusing to run.')
    if len(p)!=624 or p.partition.value_counts().get('test',0)!=505 or p.partition.value_counts().get('unseen_country_test',0)!=119:
        raise ValueError('Step 15 frozen prediction cohort size differs from documented 505+119.')
    if p[MODELS+['y']].isna().any().any(): raise ValueError('Frozen score/label missing.')
    if not p.y.isin([0,1]).all(): raise ValueError('Invalid test labels.')

    joined=p.merge(c[['Country','Year','Measles_reported_cases','Incidence_per_million',
                      'prior_3y_median_per_million','outcome_eligible','surge_C_2x_5pm']],
                   on=['Country','Year'],how='left',validate='one_to_one',indicator=True)
    if not (joined._merge=='both').all(): raise ValueError('Some frozen prediction rows are missing from candidate data.')
    if not (pd.to_numeric(joined.surge_C_2x_5pm)==joined.y).all():
        raise ValueError('Frozen target does not match candidate Definition C. STOP.')
    joined=joined.drop(columns=['_merge'])
    if not (joined.outcome_eligible.astype(str).str.lower()=='true').all():
        raise ValueError('Frozen predictions include non-eligible outcome rows.')

    # Historical WHO reporting completeness *proxy*, using only t-1..t-6.
    # Reporting observed != reliable surveillance; annual history is incomplete before 2006.
    hist=c[['Country','Year','Measles_reported_cases']].copy()
    hist['has_case_report']=hist.Measles_reported_cases.notna()
    for lag in range(1,7):
        shifted=hist[['Country','Year','has_case_report']].copy()
        shifted['Year'] += lag
        shifted=shifted.rename(columns={'has_case_report':f'case_report_lag{lag}'})
        joined=joined.merge(shifted,on=['Country','Year'],how='left',validate='one_to_one')
    # Annual WHO reporting presence in earlier history; prior three years are complete by construction
    prior_cols=[f'case_report_lag{i}' for i in range(1,7)]
    joined['prior_6y_reports_observed']=joined[prior_cols].notna().all(axis=1)
    joined['prior_6y_reporting_fraction']=joined[prior_cols].astype(float).mean(axis=1)
    joined.loc[~joined.prior_6y_reports_observed,'prior_6y_reporting_fraction']=np.nan
    joined['historical_reporting_proxy']=np.select(
        [~joined.prior_6y_reports_observed,joined.prior_6y_reporting_fraction.eq(1)],
        ['unknown','six_of_six_reported'],default='one_or_more_gaps')
    joined['vaccination_availability']=np.select(
        [joined.mcv1_lag1.notna() & joined.mcv2_lag1.notna(),
         joined.mcv1_lag1.isna() & joined.mcv2_lag1.isna()],
        ['both_available','both_missing'],default='one_missing')
    joined['high_national_coverage']=np.select(
        [joined.mcv1_lag1.isna() | joined.mcv2_lag1.isna(),
         (joined.mcv1_lag1>=95)&(joined.mcv2_lag1>=95)],
        ['unknown','both_at_least_95'],default='not_both_95')
    # COVID years can affect lagged history of target year, not a causal design.
    joined['covid_year_in_prior_3y']=joined.Year.isin([2022,2023,2024])
    joined['zero_prior_median']=joined.prior_3y_median_per_million.eq(0)

    # Frozen original ranking preserved. Mark original annual top K first;
    # subgroup selection rates are not re-ranked within each subgroup.
    selection=[]
    for part, sub in joined.groupby('partition'):
        for model in MODELS:
            selection.append(select_annual(sub,model,10))
    sel=pd.concat(selection,ignore_index=True)
    selected=joined[['partition','Country','Year']].copy()
    for model in MODELS:
        marker=sel.loc[sel.approach==model,['partition','Country','Year']].assign(selected=True)
        selected=selected.merge(marker,on=['partition','Country','Year'],how='left',validate='one_to_one')
        selected=selected.rename(columns={'selected':f'sel10_{model}'})
        selected[f'sel10_{model}']=selected[f'sel10_{model}'].eq(True)
    joined=joined.merge(selected,on=['partition','Country','Year'],validate='one_to_one')

    # Verify known primary counts before generating reports.
    expected={'test':{'rf_regularized':43,'coverage_shortfall_rank':19},
              'unseen_country_test':{'rf_regularized':8,'coverage_shortfall_rank':5}}
    for part, models in expected.items():
        sub=joined.loc[joined.partition==part]
        for model, count in models.items():
            actual=int(sub.loc[sub[f'sel10_{model}'],'y'].sum())
            if actual!=count: raise ValueError(f'Frozen {part}/{model} selected positives {actual} != expected {count}; STOP.')

    OUT.mkdir(parents=True,exist_ok=True)
    # Summary includes small subgroup Ns, not inferential claims.
    groupings=[['partition','vaccination_availability'],
               ['partition','high_national_coverage'],
               ['partition','historical_reporting_proxy'],
               ['partition','covid_year_in_prior_3y'],
               ['partition','zero_prior_median'],
               ['partition','Year']]
    results=[]
    for grouping in groupings:
        ss=summary(joined,grouping)
        if ss.empty: continue
        ss['grouping']=' + '.join(grouping)
        ss['subgroup']=ss[grouping[-1]].astype(str)
        for model in ['coverage_shortfall_rank','rf_regularized','xgb_regularized']:
            records=[]
            for keys, sub in joined.groupby(grouping,dropna=False,sort=True):
                if not isinstance(keys,tuple):keys=(keys,)
                select=sub[f'sel10_{model}']
                events=int(sub.y.sum()); detected=int(sub.loc[select,'y'].sum()); nsel=int(select.sum())
                records.append(dict(zip(grouping,keys),**{f'{model}_selected':nsel,
                            f'{model}_detected':detected,
                            f'{model}_recall':detected/events if events else np.nan,
                            f'{model}_false_alarms':nsel-detected}))
            ss=ss.merge(pd.DataFrame(records),on=grouping,validate='one_to_one')
        results.append(ss)
    pd.concat(results,ignore_index=True).to_csv(OUT/'subgroup_sensitivity_summary.csv',index=False)
    # Reporting proxy can be correlated with case history and outcome eligibility.
    # A single WHO download vintage cannot show WHO reporting delays or revisions.
    joined[['partition','Country','Year','y','vaccination_availability',
            'high_national_coverage','historical_reporting_proxy',
            'prior_6y_reporting_fraction','covid_year_in_prior_3y','zero_prior_median',
            *[f'sel10_{m}' for m in MODELS]]].to_csv(OUT/'row_level_subgroup_flags.csv',index=False)

    with open(OUT/'surveillance_sensitivity_notes.txt','w') as f:
        f.write('STEP 18: POST-TEST EXPLORATORY SURVEILLANCE / MISSINGNESS SENSITIVITY\n')
        f.write('Only frozen test predictions and prior source-derived variables used; no model retraining/tuning.\n')
        f.write('Subgroup recall uses the ORIGINAL annual top-10% country selections, NOT subgroup-specific reranking.\n')
        f.write('Historical reporting completeness proxy = WHO case value PRESENT in each of prior 6 annual records.\n')
        f.write('This is NOT verified completeness, sensitivity, case ascertainment, or surveillance accuracy.\n')
        f.write('Only annual WHO reporting, not actual outbreak confirmation; missing != zero.\n')
        f.write('COVID-overlap means 2020 or 2021 falls in the prior 3 calendar years. It is NOT causal inference.\n')
        f.write('WHO/WUENIC revisions and what was known historically CANNOT be measured with one vintage.\n')
        f.write('Small subgroups and clustered years limit inference. This analysis is post-test exploratory.\n')
        f.write('Eligibility/rankings depend retrospectively on observed target years, not feasible prospective deployment.\n')
        f.write('National WUENIC MCV1+MCV2 >=95% does NOT measure immunity or local vaccine gaps.\n')
        f.write('Pre-2022 predictions are NEVER loaded. Frozen model probabilities unchanged.\n')
    print('\nSTEP 18: SURVEILLANCE / MISSINGNESS SENSITIVITY (POST-TEST ONLY)')
    print('Frozen test predictions:',len(joined))
    print('\nVaccination missingness by test partition:')
    print(joined.groupby(['partition','vaccination_availability'],observed=True).agg(n=('y','size'),events=('y','sum')).to_string())
    print('\nHistorical reporting proxy:')
    print(joined.groupby(['partition','historical_reporting_proxy'],observed=True).agg(n=('y','size'),events=('y','sum')).to_string())
    print('\nReports:',OUT.relative_to(ROOT))
    print('NOTE: These are observational PROXY comparisons, NOT direct surveillance-quality measurements.')

if __name__=='__main__':
    main()
