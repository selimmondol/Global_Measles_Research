#!/usr/bin/env python3
"""Step 11: validation-only reliability audit for Measles ML models.

Run at project root:
    python src/11_advanced_ml_reliability.py

Inputs:
    results/advanced_ml_models/validation_predictions.csv
    results/baseline_models/validation_predictions.csv
    data/processed/Measles_Model_Features.csv (ONLY validation-year vaccination features)

Outputs: results/advanced_ml_reliability/*.csv, calibration_curve.png, reliability_notes.txt

The 2022-2025 test set and unseen-country test are never accessed.
No models are refit. Target is a provisional annual *reported-incidence surge*,
not a WHO-defined measles outbreak.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
ML_FILE = ROOT / 'results/advanced_ml_models/validation_predictions.csv'
BASE_FILE = ROOT / 'results/baseline_models/validation_predictions.csv'
FEATURE_FILE = ROOT / 'data/processed/Measles_Model_Features.csv'
OUT = ROOT / 'results/advanced_ml_reliability'
COHORTS = ('matched_complete_vaccination', 'all_eligible_imputed_vaccination')
MODELS = ('combined', 'rf_regularized', 'rf_shallow', 'xgb_regularized', 'xgb_shallow')
PAIRS = (('rf_regularized','combined'),('xgb_regularized','combined'),
         ('rf_regularized','xgb_regularized'))
METRICS = ('AUROC', 'average_precision', 'brier_score', 'recall_at_10pct')
B = 500
SEED = 42

def score(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) < 3 or np.unique(y).size < 2 or not np.isfinite(p).all():
        return None
    k = max(1, int(np.ceil(.10 * len(y))))
    order = np.argsort(-p, kind='stable')[:k]
    tp = int(y[order].sum())
    return {'n':len(y), 'events':int(y.sum()),
            'AUROC':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'brier_score':float(brier_score_loss(y,p)),
            'recall_at_10pct':float(tp/y.sum()),
            'precision_at_10pct':float(tp/k),
            'top10_n':k, 'top10_true_positive':tp,
            'top10_false_positive':int(k-tp),
            'top10_missed_events':int(y.sum()-tp)}

def load_predictions(path, models):
    if not path.exists():
        raise SystemExit(f'Missing {path}; run Steps 8 and 10 first.')
    d = pd.read_csv(path, low_memory=False)
    cols = {'cohort','model','Country','Year','observed_surge_label','predicted_probability'}
    if cols - set(d):
        raise SystemExit(f'{path.name} missing columns: {sorted(cols-set(d))}')
    # Only the named models are needed; other source models must not affect comparison.
    d = d.loc[d['model'].isin(models),list(cols)].copy()
    if d.empty or not d.Year.between(2019,2021).all():
        raise SystemExit('STOP: non-validation year in predictions (allowed 2019–2021 only)')
    if not d.cohort.isin(COHORTS).all():
        raise SystemExit('STOP: Unexpected cohort')
    if d.duplicated(['cohort','model','Country','Year']).any():
        raise SystemExit('STOP: Duplicate prediction keys')
    if d[['observed_surge_label','predicted_probability']].isna().any().any():
        raise SystemExit('STOP: Missing label or prediction')
    if not d.observed_surge_label.isin([0,1]).all() or not d.predicted_probability.between(0,1).all():
        raise SystemExit('STOP: Invalid label or probability')
    return d

def main():
    ml = load_predictions(ML_FILE, MODELS[1:])
    baseline = load_predictions(BASE_FILE, ('combined',))
    d = pd.concat([ml,baseline], ignore_index=True)
    if d.duplicated(['cohort','model','Country','Year']).any():
        raise SystemExit('STOP: duplicate model/cohort/country/year across inputs')
    if not FEATURE_FILE.exists():
        raise SystemExit(f'Missing {FEATURE_FILE}. Run Step 6 first.')
    f = pd.read_csv(FEATURE_FILE, usecols=['Country','Year','mcv1_lag1','mcv2_lag1'])
    f = f.loc[f.Year.between(2019,2021)].copy()  # do not join any future years
    if f.duplicated(['Country','Year']).any():
        raise SystemExit('STOP: Duplicate feature keys')
    d = d.merge(f, on=['Country','Year'], how='left', validate='many_to_one', indicator=True)
    if not d['_merge'].eq('both').all():
        raise SystemExit('STOP: Unmatched validation feature keys')
    d = d.drop(columns='_merge')
    d['both_vaccines_known'] = d[['mcv1_lag1','mcv2_lag1']].notna().all(axis=1)
    d['high_vax_prior_year'] = d['both_vaccines_known'] & d.mcv1_lag1.ge(95) & d.mcv2_lag1.ge(95)

    metrics, ci_rows, differences = [], [], []
    alignment, subgroup_rows, calibration, missed = [], [], [], []
    rng = np.random.default_rng(SEED)
    for cohort in COHORTS:
        g = d.loc[d.cohort.eq(cohort)].copy()
        frames = {}
        for model in MODELS:
            one = g.loc[g.model.eq(model)].sort_values(['Country','Year']).reset_index(drop=True)
            if one.empty:
                raise SystemExit(f'STOP: Missing model {model}, cohort {cohort}')
            frames[model] = one
        ref = frames['combined']
        key_cols = ['Country','Year','observed_surge_label']
        for model in MODELS[1:]:
            if not ref[key_cols].equals(frames[model][key_cols]):
                raise SystemExit(f'STOP: Validation keys or target labels differ: {cohort}/{model}')
        if cohort == 'matched_complete_vaccination' and not ref.both_vaccines_known.all():
            raise SystemExit('STOP: Vaccination missing in matched-complete cohort')
        n = len(ref)
        y = ref.observed_surge_label.to_numpy(dtype=int)
        countries = ref.Country.to_numpy(dtype=str)
        population = np.unique(countries)
        pos = {country:np.flatnonzero(countries == country) for country in population}
        probabilities = {m:frames[m].predicted_probability.to_numpy(dtype=float) for m in MODELS}
        alignment.append({'cohort':cohort,'n':n,'events':int(y.sum()),
                          'unique_countries':len(population),'same_keys_and_labels_all_models':True})
        for model in MODELS:
            one = frames[model]
            p = probabilities[model]
            s = score(y,p)
            if s is None:
                raise SystemExit('STOP: Validation has fewer than 2 classes')
            metrics.append({'cohort':cohort,'model':model,**s})
            # Fixed-width bins: show n, never hide small/sparse bins.
            bins = pd.cut(p, bins=[0,.2,.4,.6,.8,1.0],include_lowest=True)
            for interval in bins.categories:
                mask = np.asarray(bins == interval)
                calibration.append({'cohort':cohort,'model':model,'probability_bin':str(interval),
                                    'n':int(mask.sum()),
                                    'mean_predicted':float(p[mask].mean()) if mask.any() else np.nan,
                                    'observed_event_rate':float(y[mask].mean()) if mask.any() else np.nan})
            for subgroup, selector in (
                ('high_coverage_both_doses_ge95',one.high_vax_prior_year),
                ('other_observed_coverage',one.both_vaccines_known & ~one.high_vax_prior_year),
                ('missing_vaccination',~one.both_vaccines_known)):
                mask=selector.to_numpy(dtype=bool)
                sm=score(y[mask],p[mask])
                subgroup_rows.append({'cohort':cohort,'model':model,'subgroup':subgroup,
                                      'n':int(mask.sum()),'events':int(y[mask].sum()),
                                      **({k:v for k,v in (sm or {}).items() if k not in ('n','events')})})
            # Ranking on country-YEAR rows; multiple rows may represent the same country.
            k=max(1,int(np.ceil(.10*n)))
            top=one.sort_values(['predicted_probability','Country','Year'],ascending=[False,True,True]).head(k)
            selected=set(zip(top.Country,top.Year))
            for r in one.loc[one.observed_surge_label.eq(1)].itertuples(index=False):
                if (r.Country,r.Year) not in selected:
                    missed.append({'cohort':cohort,'model':model,'Country':r.Country,
                                   'Year':r.Year,'predicted_probability':r.predicted_probability,
                                   'high_vax_prior_year':bool(r.high_vax_prior_year)})

        # Paired country cluster bootstrap: resample whole countries and retain their years.
        dist = {m:{metric:[] for metric in METRICS} for m in MODELS}
        paired={pair:{metric:[] for metric in METRICS} for pair in PAIRS}
        valid=0
        for _ in range(B):
            draw=rng.choice(population,size=len(population),replace=True)
            idx=np.concatenate([pos[c] for c in draw])
            yy=y[idx]
            if np.unique(yy).size < 2:
                continue
            scored={m:score(yy,probabilities[m][idx]) for m in MODELS}
            if any(v is None for v in scored.values()):
                continue
            valid+=1
            for m in MODELS:
                for metric in METRICS:
                    dist[m][metric].append(scored[m][metric])
            for pair in PAIRS:
                for metric in METRICS:
                    paired[pair][metric].append(scored[pair[0]][metric]-scored[pair[1]][metric])
        if valid < 100:
            raise SystemExit(f'Only {valid} usable bootstrap draws; cannot report stable CI')
        for m in MODELS:
            for metric in METRICS:
                v=np.asarray(dist[m][metric])
                ci_rows.append({'cohort':cohort,'model':m,'metric':metric,
                                'estimate':score(y,probabilities[m])[metric],
                                'lower_95pct':np.quantile(v,.025), 'upper_95pct':np.quantile(v,.975),
                                'n_boot_valid':valid,'n_boot_requested':B})
        for a,b in PAIRS:
            for metric in METRICS:
                v=np.asarray(paired[(a,b)][metric])
                diffs=score(y,probabilities[a])[metric]-score(y,probabilities[b])[metric]
                differences.append({'cohort':cohort,'comparison':f'{a}_minus_{b}',
                                    'metric':metric,'difference':diffs,
                                    'lower_95pct':np.quantile(v,.025),'upper_95pct':np.quantile(v,.975),
                                    'n_boot_valid':valid})
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(OUT/'validation_metrics_recomputed.csv',index=False)
    pd.DataFrame(ci_rows).to_csv(OUT/'bootstrap_intervals.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'paired_model_differences.csv',index=False)
    pd.DataFrame(subgroup_rows).to_csv(OUT/'high_vaccination_subgroup.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'calibration_bins.csv',index=False)
    pd.DataFrame(missed).to_csv(OUT/'missed_events.csv',index=False)
    pd.DataFrame(alignment).to_csv(OUT/'cohort_alignment.csv',index=False)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax=plt.subplots(figsize=(7,5))
        z=pd.DataFrame(calibration)
        for m in ('combined','rf_regularized','xgb_regularized'):
            t=z[(z.cohort=='matched_complete_vaccination')&(z.model==m)&(z.n>0)]
            ax.plot(t.mean_predicted,t.observed_event_rate,marker='o',label=m)
        ax.plot([0,1],[0,1],linestyle='--',label='ideal calibration')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',
               ylabel='Observed positive fraction',
               title='Validation calibration (matched cohort)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT/'calibration_curve.png',dpi=180)
        plt.close(fig)
    except ImportError:
        print('matplotlib not installed; skipping calibration_curve.png')
    (OUT/'reliability_notes.txt').write_text(
        'STEP 11: ADVANCED ML VALIDATION RELIABILITY AUDIT\n'
        'Only 2019–2021 validation predictions loaded; no test partitions evaluated.\n'
        'Models: combined Logistic Regression; four prespecified RF/XGBoost configurations.\n'
        'All models aligned to identical Country-Year and observed outcome for each cohort.\n'
        'Paired percentile confidence intervals use 500 country-cluster bootstrap draws.\n'
        'Country clustering does not account fully for global calendar-year shocks.\n'
        'Repeated comparison and configuration selection on validation induces optimism.\n'
        'Top 10% is a country-YEAR ranking, not annual distinct-country prioritization.\n'
        'Calibration bins may be sparsely populated; check n before interpreting curves.\n'
        'High vaccination is NATIONAL lagged WUENIC >=95% for both doses, not immunity.\n'
        'Historical WUENIC revisions are not historical real-time data vintages.\n'
        'Surge C is a provisional researcher-defined reported-incidence outcome, NOT a WHO outbreak.\n'
        'No model refitting, feature selection, test-set evaluation or raw data overwrite.\n',
        encoding='utf-8')
    print('\nSTEP 11: ADVANCED ML RELIABILITY (VALIDATION ONLY)\n')
    print(pd.DataFrame(metrics)[['cohort','model','n','events','AUROC','average_precision',
                                  'brier_score','recall_at_10pct']].round(4).to_string(index=False))
    print('\nPaired bootstrap comparisons:')
    print(pd.DataFrame(differences).query("cohort == 'all_eligible_imputed_vaccination' and metric == 'AUROC'")
          [['comparison','difference','lower_95pct','upper_95pct']].round(4).to_string(index=False))
    print('\nReports:',OUT.relative_to(ROOT))
    print('NOTE: Future-year and unseen-country test partitions were NOT used.')

if __name__ == '__main__':
    main()
