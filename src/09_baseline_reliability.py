#!/usr/bin/env python3
"""Step 9: Validation-only reliability audit for Step 8 logistic baselines.

Run at project root: python src/09_baseline_reliability.py

Inputs:
 results/baseline_models/validation_predictions.csv
 data/processed/Measles_Model_Features.csv (validation vaccination indicators ONLY)

Outputs in results/baseline_reliability/:
 metrics_recomputed.csv, bootstrap_intervals.csv, paired_model_differences.csv,
 calibration_bins.csv, high_vaccination_subgroup.csv, missed_events.csv,
 cohort_alignment.csv, reliability_notes.txt, calibration_curve.png

Only 2019-2021 validation observations are evaluated. Future and unseen-country
TEST results are not accessed. No source data or fitted model is modified.
This is descriptive/exploratory: outcome C is researcher-defined, not WHO outbreak.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / 'results/baseline_models/validation_predictions.csv'
FEAT = ROOT / 'data/processed/Measles_Model_Features.csv'
OUT = ROOT / 'results/baseline_reliability'
SEED = 42
N_BOOT = 500
MODEL_ORDER = ['vaccination_only', 'incidence_history', 'combined']
COHORT_ORDER = ['matched_complete_vaccination', 'all_eligible_imputed_vaccination']


def evaluate(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) < 3 or len(np.unique(y)) != 2:
        return None
    k = max(1, int(np.ceil(.10 * len(y))))
    ranking = np.argsort(-p, kind='stable')[:k]
    return {'AUROC': roc_auc_score(y, p),
            'average_precision': average_precision_score(y, p),
            'brier_score': brier_score_loss(y, p),
            'recall_at_10pct': y[ranking].sum() / y.sum(),
            'precision_at_10pct': y[ranking].mean(),
            'top10_n': k, 'top10_true_positive': int(y[ranking].sum()),
            'top10_false_positive': int(k - y[ranking].sum()),
            'top10_missed_events': int(y.sum() - y[ranking].sum()),
            'mean_predicted_risk': p.mean(), 'observed_event_rate': y.mean()}


def main():
    for path in (PRED, FEAT):
        if not path.is_file():
            raise SystemExit(f'Missing input: {path}. Run Step 8 first.')
    d = pd.read_csv(PRED, low_memory=False)
    required = {'cohort','model','Country','Year','observed_surge_label','predicted_probability'}
    if required - set(d):
        raise SystemExit('Prediction input lacks: ' + ', '.join(sorted(required - set(d))))
    if not d['Year'].between(2019, 2021).all():
        raise SystemExit('STOP: file contains years outside VALIDATION 2019-2021. Test data not allowed.')
    if d.duplicated(['cohort','model','Country','Year']).any():
        raise SystemExit('Duplicate predictions in cohort/model/country/year')
    if d['predicted_probability'].isna().any() or not d['predicted_probability'].between(0,1).all():
        raise SystemExit('Missing or invalid predicted probability')
    if d['observed_surge_label'].isna().any() or not d['observed_surge_label'].isin([0,1]).all():
        raise SystemExit('Invalid binary labels')
    if set(d.model) != set(MODEL_ORDER) or set(d.cohort) != set(COHORT_ORDER):
        raise SystemExit('Unexpected Step 8 models/cohorts. Audit script before continuing.')

    # Load features ONLY for vaccination subgroup definitions; select validation
    # rows, country/year and the two LAGGED doses; do not access outcome or tests.
    f = pd.read_csv(FEAT, usecols=['Country','Year','mcv1_lag1','mcv2_lag1'])
    f = f[f.Year.between(2019,2021)].copy()
    if f.duplicated(['Country','Year']).any():
        raise SystemExit('Duplicate validation vaccination rows')
    d = d.merge(f, on=['Country','Year'], how='left', validate='many_to_one', indicator=True)
    if not d['_merge'].eq('both').all():
        raise SystemExit('Some validation predictions have no matching feature row')
    d = d.drop(columns=['_merge'])
    d['high_vax_prior_year'] = d.mcv1_lag1.ge(95) & d.mcv2_lag1.ge(95)
    d['both_vax_known'] = d[['mcv1_lag1','mcv2_lag1']].notna().all(axis=1)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_rows, calibration_rows, subgroup_rows, missed_rows, checks = [], [], [], [], []
    rng = np.random.default_rng(SEED)
    bootstrap_rows, difference_rows = [], []

    for cohort in COHORT_ORDER:
        cd = d[d.cohort.eq(cohort)].copy()
        frames = {}
        keys_reference = None
        for model in MODEL_ORDER:
            g = cd[cd.model.eq(model)].reset_index(drop=True)
            keys = list(map(tuple, g[['Country','Year']].itertuples(index=False, name=None)))
            if keys_reference is None:
                keys_reference = keys
            elif keys != keys_reference:
                raise SystemExit(f'Fair comparison failed: key mismatch in {cohort}/{model}')
            frames[model] = g
            m = evaluate(g.observed_surge_label, g.predicted_probability)
            if m is None:
                raise SystemExit('Only one outcome class in validation cohort')
            metrics_rows.append({'cohort':cohort,'model':model,'n':len(g),
                                 'events':int(g.observed_surge_label.sum()), **m})
            # Fixed [0,0.2), ..., [0.8,1] probability bins (not adaptive).
            bins = pd.cut(g.predicted_probability, bins=[0,.2,.4,.6,.8,1],
                          include_lowest=True)
            for bucket, piece in g.groupby(bins, observed=False):
                calibration_rows.append({'cohort':cohort,'model':model,'probability_bin':str(bucket),
                                         'n':len(piece),'mean_predicted':piece.predicted_probability.mean(),
                                         'observed_event_rate':piece.observed_surge_label.mean()})
            for label, sub in [('high_coverage_both_doses_ge95',g[g.high_vax_prior_year]),
                               ('other_observed_coverage',g[g.both_vax_known & ~g.high_vax_prior_year]),
                               ('missing_vaccination',g[~g.both_vax_known])]:
                x = evaluate(sub.observed_surge_label, sub.predicted_probability)
                subgroup_rows.append({'cohort':cohort,'model':model,'subgroup':label,
                                      'n':len(sub),'events':int(sub.observed_surge_label.sum()),
                                      **(x if x is not None else {})})
            # Top 10% capacity: rows represent country-years, NOT unique countries.
            k = max(1, int(np.ceil(.1*len(g))))
            ranked = g.sort_values(['predicted_probability','Country','Year'],
                                   ascending=[False,True,True]).copy()
            selected = set(zip(ranked.iloc[:k].Country, ranked.iloc[:k].Year))
            missed = g[g.observed_surge_label.eq(1) &
                       ~pd.Series(list(zip(g.Country,g.Year)),index=g.index).isin(selected)]
            for r in missed.itertuples(index=False):
                missed_rows.append({'cohort':cohort,'model':model,'Country':r.Country,'Year':r.Year,
                                    'predicted_probability':r.predicted_probability,
                                    'high_vax_prior_year':bool(r.high_vax_prior_year)})
        y = frames['combined'].observed_surge_label.to_numpy(dtype=int)
        countries = frames['combined'].Country.to_numpy()
        unique_countries = np.unique(countries)
        checks.append({'cohort':cohort,'validation_rows':len(y),'validation_countries':len(unique_countries),
                       'events':int(y.sum()),'aligned_models':True})
        p_array = {name:frames[name].predicted_probability.to_numpy(dtype=float) for name in MODEL_ORDER}
        position_groups = {c:np.flatnonzero(countries == c) for c in unique_countries}
        draws = {name: [] for name in MODEL_ORDER}
        paired = {name: [] for name in ['AUROC','average_precision','brier_score','recall_at_10pct']}
        used = 0
        for _ in range(N_BOOT):
            sampled = rng.choice(unique_countries,size=len(unique_countries),replace=True)
            idx = np.concatenate([position_groups[c] for c in sampled])
            yb = y[idx]
            if np.unique(yb).size < 2:
                continue
            bm = {name:evaluate(yb, p_array[name][idx]) for name in MODEL_ORDER}
            if any(v is None for v in bm.values()):
                continue
            used += 1
            for name in MODEL_ORDER:
                draws[name].append(bm[name])
            for metric in paired:
                paired[metric].append(bm['combined'][metric]-bm['vaccination_only'][metric])
        for name in MODEL_ORDER:
            for metric in ['AUROC','average_precision','brier_score','recall_at_10pct']:
                samples = [v[metric] for v in draws[name]]
                bootstrap_rows.append({'cohort':cohort,'model':name,'metric':metric,
                                       'n_boot_requested':N_BOOT,'n_boot_valid':used,
                                       'lower_95pct':float(np.quantile(samples,.025)),
                                       'upper_95pct':float(np.quantile(samples,.975))})
        for metric, values in paired.items():
            difference_rows.append({'cohort':cohort,'comparison':'combined_minus_vaccination_only',
                                    'metric':metric,'n_boot_valid':used,
                                    'lower_95pct':float(np.quantile(values,.025)),
                                    'upper_95pct':float(np.quantile(values,.975))})

    pd.DataFrame(metrics_rows).to_csv(OUT/'metrics_recomputed.csv',index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT/'calibration_bins.csv',index=False)
    pd.DataFrame(subgroup_rows).to_csv(OUT/'high_vaccination_subgroup.csv',index=False)
    pd.DataFrame(missed_rows).to_csv(OUT/'missed_events.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'cohort_alignment.csv',index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT/'bootstrap_intervals.csv',index=False)
    pd.DataFrame(difference_rows).to_csv(OUT/'paired_model_differences.csv',index=False)

    # Optional descriptive figure without altering evaluation data.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,5))
        for model in MODEL_ORDER:
            z = pd.DataFrame(calibration_rows)
            z = z[(z.cohort == 'matched_complete_vaccination') & (z.model == model) & z.n.gt(0)]
            ax.plot(z.mean_predicted,z.observed_event_rate,marker='o',label=model)
        ax.plot([0,1],[0,1],linestyle='--',label='ideal calibration')
        ax.set(xlabel='Mean predicted probability',ylabel='Observed fraction positive',
               title='Baseline calibration (validation; matched cohort)',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT/'calibration_curve.png',dpi=180)
        plt.close(fig)
    except ImportError:
        print('NOTE: matplotlib unavailable; CSV calibration tables still produced.')

    (OUT/'reliability_notes.txt').write_text(
        'STEP 9: VALIDATION-ONLY BASELINE RELIABILITY AUDIT\n'
        'Target C is researcher-defined annual reported-incidence surge, NOT confirmed measles outbreaks.\n'
        'Only 2019-2021 validation predictions; no test data or test predictions loaded.\n'
        'Bootstrap resamples entire countries (with replacement), retaining their years.\n'
        'Intervals are percentile 95% descriptive confidence intervals, not causal intervals.\n'
        'All models compared on identical country-years within each cohort.\n'
        'Recall@10% selects top 10% of country-YEAR rows; repeated countries possible.\n'
        'Calibration bins are fixed 0.2-width; small bins are unstable.\n'
        'High coverage: national, prior-year WUENIC MCV1 AND MCV2 >=95%; not immunity.\n'
        'WUENIC current-revision values may not have been available at prediction time.\n'
        'Missing coverage subgroup is descriptive and may be nonrandom.\n'
        'Do not select models or operating points using any future/unseen-country test.\n'
        'Bootstrap confidence intervals alone do not adjust for all global time shocks.\n', encoding='utf-8')
    print('\nSTEP 9: BASELINE RELIABILITY (VALIDATION ONLY)\n')
    print(pd.DataFrame(metrics_rows)[['cohort','model','n','events','AUROC','average_precision',
                                      'brier_score','recall_at_10pct']].round(4).to_string(index=False))
    print('\nCountry-cluster bootstrap draws:',N_BOOT)
    print('Saved reports:',OUT.relative_to(ROOT))
    print('NOTE: Future-year and unseen-country tests were not used.')

if __name__ == '__main__':
    main()
