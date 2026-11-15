#!/usr/bin/env python3
"""Step 19: READ-ONLY publication tables, figures, and calibration diagnostics.
Run: python src/19_publication_results.py
Reads frozen Step 15 predictions and Steps 16-18 reports. Does not train,
recalibrate, optimize, change outcomes, or overwrite previous analyses.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
S15 = ROOT / 'results' / 'independent_test_v1'
S16 = ROOT / 'results' / 'independent_test_reliability_v1'
S17 = ROOT / 'results' / 'outcome_robustness_posttest_v1'
S18 = ROOT / 'results' / 'surveillance_sensitivity_posttest_v1'
OUT = ROOT / 'results' / 'publication_results_v1'
MODELS = ['vaccination_only', 'combined', 'rf_regularized', 'xgb_regularized']
PARTITIONS = ['test', 'unseen_country_test']
NAMES = {'vaccination_only': 'Vaccination-only LR', 'combined': 'Combined LR',
         'rf_regularized': 'Random Forest', 'xgb_regularized': 'XGBoost',
         'coverage_shortfall_rank': 'Coverage-shortfall ranking'}

def must_read(base, name, required):
    p = base / name
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}\nRun the earlier step first; existing data will not be reconstructed.')
    df = pd.read_csv(p)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{p}: missing columns {sorted(missing)}')
    return df

def save(df, filename):
    df.to_csv(OUT / filename, index=False, float_format='%.8g')

def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches='tight')
    plt.close()

def main():
    pred = must_read(S15,'test_predictions.csv', ['partition','Country','Year','y']+MODELS)
    cls = must_read(S15,'test_classification_metrics.csv',
                    ['partition','model','n','events','AUROC','average_precision','brier_score'])
    annual = must_read(S15,'annual_decision_metrics.csv',
                       ['partition','Year','approach','capacity_pct','evaluated','selected','events','detected','recall'])
    pooled = must_read(S15,'pooled_decision_summary.csv',
                       ['partition','approach','capacity_pct','selected','events','detected','recall','precision'])
    ci = must_read(S16,'paired_bootstrap_differences.csv',
                   ['partition','model_a','observed_difference','lower_95pct','upper_95pct'])
    rob = must_read(S17,'decision_metrics_by_outcome.csv',
                    ['outcome','partition','approach','capacity_pct','events','selected','detected','recall'])
    subgroup = must_read(S18,'subgroup_sensitivity_summary.csv',
                         ['partition','grouping','subgroup','n','events','rf_regularized_detected',
                          'rf_regularized_recall','coverage_shortfall_rank_detected'])
    receipt_path = S15 / 'evaluation_receipt.json'
    if not receipt_path.exists(): raise FileNotFoundError(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if int(receipt.get('frozen_manifest_files_checked',-1)) != 28:
        raise ValueError('Unexpected freeze check in Step 15 receipt')
    if pred.duplicated(['partition','Country','Year']).any():
        raise ValueError('Duplicate prediction Country-Year within a partition')
    if set(pred.partition.unique()) != set(PARTITIONS):
        raise ValueError('Unexpected test partitions')
    if not pred.y.isin([0,1]).all(): raise ValueError('Outcome must be binary')
    for p,n,e in [('test',505,143),('unseen_country_test',119,33)]:
        d=pred[pred.partition==p]
        if len(d)!=n or int(d.y.sum())!=e:
            raise ValueError(f'{p}: unexpected cohort size or events; aborting')
    for m in MODELS:
        if (pred[m].isna().any() or not pred[m].between(0,1).all()):
            raise ValueError(f'Bad probability score in {m}')
    # Independently check published score metrics, without modifying them.
    audit=[]
    for part,d in pred.groupby('partition',sort=False):
        for m in MODELS:
            row=cls[(cls.partition==part)&(cls.model==m)]
            if len(row)!=1: raise ValueError(f'Missing score row {part}/{m}')
            au=roc_auc_score(d.y,d[m]); ap=average_precision_score(d.y,d[m]); br=brier_score_loss(d.y,d[m])
            if not all(abs(float(row.iloc[0][k])-v)<0.00015 for k,v in
                       [('AUROC',au),('average_precision',ap),('brier_score',br)]):
                raise ValueError(f'Mismatch with Step 15 score metrics: {part}/{m}')
            audit.append({'partition':part,'model':m,'n':len(d),'events':int(d.y.sum()),
                          'AUROC':au,'average_precision':ap,'brier_score':br,
                          'mean_predicted_risk':d[m].mean(),'event_rate':d.y.mean(),
                          'observed_minus_predicted':d.y.mean()-d[m].mean()})
    OUT.mkdir(parents=True, exist_ok=True)
    audit=pd.DataFrame(audit)
    save(audit,'table1_model_performance.csv')
    primary=pooled[pooled.capacity_pct.isin([10,20])].copy()
    primary['method_display']=primary.approach.map(NAMES)
    save(primary,'table2_annual_decision_performance.csv')
    ann10=annual[annual.capacity_pct.eq(10)].copy()
    save(ann10,'table3_year_by_year_recall.csv')
    ci10=ci.copy()
    if 'capacity_pct' in ci10.columns: ci10=ci10[ci10.capacity_pct.eq(10)]
    if 'model_b' in ci10.columns:
        ci10=ci10[ci10.model_b.eq('coverage_shortfall_rank')]
    if 'metric' in ci10.columns:
        ci10=ci10[ci10.metric.astype(str).str.contains('recall',case=False)]
    save(ci10,'table4_paired_recall_bootstrap.csv')
    robustness=rob[rob.capacity_pct.eq(10) & rob.approach.isin(
        ['coverage_shortfall_rank','rf_regularized','xgb_regularized'])].copy()
    save(robustness,'table5_outcome_robustness_exploratory.csv')
    groups=['partition + vaccination_availability','partition + historical_reporting_proxy',
            'partition + high_national_coverage']
    sub=subgroup[subgroup.grouping.isin(groups)].copy()
    save(sub,'table6_surveillance_subgroups_exploratory.csv')

    # Calibration diagnostics: equal-width 10 bins; unoccupied bins omitted.
    # ECE is descriptive only; not proof of prospective calibration.
    bins=[]; cal=[]
    for (part,m),d in [( (p,m), pred[pred.partition.eq(p)]) for p in PARTITIONS for m in MODELS]:
        prob=d[m].to_numpy(float); y=d.y.to_numpy(int)
        ids=np.minimum((prob*10).astype(int),9)
        ece=0.0
        for b in range(10):
            mask=ids==b; n=int(mask.sum())
            if n:
                pm=float(prob[mask].mean()); obs=float(y[mask].mean()); ece += n/len(d)*abs(pm-obs)
            else: pm=obs=np.nan
            bins.append({'partition':part,'model':m,'bin_lower':b/10,'bin_upper':(b+1)/10,
                         'n':n,'mean_predicted':pm,'observed_rate':obs})
        cal.append({'partition':part,'model':m,'n':len(d),'events':int(y.sum()),
                    'mean_predicted':float(prob.mean()),'observed_event_rate':float(y.mean()),
                    'observed_minus_predicted':float(y.mean()-prob.mean()),
                    'brier_score':float(brier_score_loss(y,prob)),'ece_10bins_descriptive':ece})
    bins=pd.DataFrame(bins); cal=pd.DataFrame(cal)
    save(bins,'table7_calibration_bins.csv')
    save(cal,'table8_calibration_summary.csv')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharey=True)
    for ax,part in zip(axes,PARTITIONS):
        d=audit[audit.partition.eq(part)].set_index('model').loc[MODELS]
        x=np.arange(len(d)); w=.36
        ax.bar(x-w/2,d.AUROC,w,label='AUROC',color='#204c73')
        ax.bar(x+w/2,d.average_precision,w,label='Average precision',color='#8ba7bd')
        ax.set_xticks(x,[NAMES[m] for m in MODELS],rotation=25,ha='right')
        ax.set_ylim(0,1);ax.set_title('Future years (2022–2025)' if part=='test' else 'Unseen countries')
        ax.grid(axis='y',alpha=.17)
    axes[0].set_ylabel('Score (higher is better)'); axes[0].legend(fontsize=8)
    save_plot('figure1_discrimination.png')

    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharey=True)
    for ax,part in zip(axes,PARTITIONS):
        data=ann10[ann10.partition.eq(part) & ann10.approach.isin(['coverage_shortfall_rank','rf_regularized','xgb_regularized'])]
        for approach,color,style in [('coverage_shortfall_rank','#9a6b54','--'),
                                      ('rf_regularized','#204c73','-'),('xgb_regularized','#558a86',':')]:
            d=data[data.approach.eq(approach)].sort_values('Year')
            ax.plot(d.Year,d.recall*100,marker='o',linestyle=style,color=color,label=NAMES[approach])
        ax.set_xticks(sorted(data.Year.unique()));ax.grid(alpha=.2);ax.set_ylim(0,75)
        ax.set_title('Future years' if part=='test' else 'Unseen countries')
    axes[0].set_ylabel('Annual Recall@10% (%)'); axes[0].legend(fontsize=8)
    save_plot('figure2_annual_recall.png')

    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),sharex=True,sharey=True)
    for ax,part in zip(axes,PARTITIONS):
        ax.plot([0,1],[0,1],color='gray',linestyle='--',linewidth=1,label='Perfect calibration')
        for m,color in [('vaccination_only','#9a6b54'),('combined','#999d9f'),
                        ('rf_regularized','#204c73'),('xgb_regularized','#558a86')]:
            d=bins[bins.partition.eq(part) & bins.model.eq(m) & bins.n.gt(0)]
            ax.plot(d.mean_predicted,d.observed_rate,marker='o',markersize=3,
                    linewidth=1.3,color=color,label=NAMES[m])
        ax.set_title('Future years' if part=='test' else 'Unseen countries')
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.grid(alpha=.18)
        ax.set_xlabel('Mean predicted probability')
    axes[0].set_ylabel('Observed event fraction');axes[0].legend(fontsize=8)
    save_plot('figure3_calibration.png')

    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=False)
    for ax,part in zip(axes,PARTITIONS):
        d=bins[bins.partition.eq(part) & bins.model.eq('rf_regularized')]
        ax.bar((d.bin_lower+d.bin_upper)/2,d.n,width=.085,color='#204c73')
        ax.set_xticks(np.arange(.05,1.01,.1),[f'{x:.1f}' for x in np.arange(0,1,.1)],rotation=0)
        ax.set_title('Future years' if part=='test' else 'Unseen countries')
        ax.set_xlabel('RF predicted-risk bin (lower bound)');ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Observations per bin')
    save_plot('figure4_calibration_bin_support.png')

    f=ci10[ci10.model_a.isin(['rf_regularized','xgb_regularized'])].copy()
    if len(f):
        fig,ax=plt.subplots(figsize=(8,3.8))
        labels=[]
        for j,(_,r) in enumerate(f.iterrows()):
            center=100*r.observed_difference;lo=100*r.lower_95pct;hi=100*r.upper_95pct
            ax.errorbar(center,j,xerr=np.array([[center-lo],[hi-center]]),fmt='o',
                        color='#204c73',capsize=4)
            labels.append(('Future' if r.partition=='test' else 'Unseen')+' · '+NAMES[r.model_a])
        ax.axvline(0,linestyle='--',color='gray');ax.set_yticks(range(len(labels)),labels)
        ax.invert_yaxis();ax.set_xlabel('Recall@10% difference vs coverage ranking (percentage points)')
        ax.set_title('Post-test country-cluster bootstrap (descriptive)');ax.grid(axis='x',alpha=.2)
        save_plot('figure5_paired_bootstrap_intervals.png')

    txt='''STEP 19 | RESULTS PACKAGE: READ-ONLY, POST-TEST\n\n'''
    txt+='Research target: researcher-defined annual *reported-incidence surge*, NOT WHO-confirmed outbreak.\n'
    txt+='Only original Step-15 frozen predictions were used; no refitting, feature changes, threshold changes, optimization or recalibration.\n'
    txt+='The held-out outcomes were already opened in Step 15. Steps 16–19 are POST-TEST exploratory/descriptive, not new independent validation.\n'
    txt+='Calibration is calculated on pooled country-year rows with 10 equal-width predicted-risk bins. ECE is a descriptive bin-dependent diagnostic.\n'
    txt+='Small bins are unstable; see table7 and figure4. Predicted probabilities are NOT prospectively validated, and no calibration correction was applied.\n'
    txt+='Country-year outcomes are correlated; bootstrap intervals reflect country-resampling only, not common global shocks across just four calendar years.\n'
    txt+='Alternative outcome cohorts can exclude rows, changing selection counts and denominators; do not compare raw recall as if same cohort.\n'
    txt+='Strict new-onset analysis did NOT show primary-model superiority and must be reported prominently.\n'
    txt+='Vaccination/reporting subgroup analyses use proxies and small counts; missing does not mean zero.\n'
    txt+='Historic WHO/WUENIC values may have been revised since prediction year; genuine real-time forecasting is NOT demonstrated.\n'
    txt+='Retrospective outcome-evaluable ranking does not reflect prospective deployment selection.\n'
    txt+='Do not claim effects of interventions or WHO-system superiority from a simple coverage-shortfall comparator.\n'
    txt+='Do not use these figures to choose or retune the frozen winner.\n\n'
    txt+='Inputs SHA-256:\n'
    for p in [S15/'test_predictions.csv',S15/'test_classification_metrics.csv',
              S15/'annual_decision_metrics.csv',S15/'pooled_decision_summary.csv',
              S16/'paired_bootstrap_differences.csv',S17/'decision_metrics_by_outcome.csv',
              S18/'subgroup_sensitivity_summary.csv',receipt_path]:
        txt+=hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n'
    (OUT/'RESULTS_INTERPRETATION_AND_PROVENANCE.txt').write_text(txt,encoding='utf-8')
    print('STEP 19: PUBLICATION TABLES, FIGURES & CALIBRATION (READ-ONLY)')
    print('Verified:',len(pred),'frozen test rows | score alignment checked')
    print('\nModel performance:')
    print(audit[['partition','model','AUROC','average_precision','brier_score','observed_minus_predicted']].round(4).to_string(index=False))
    print('\nRecall @10%:')
    print(primary[primary.capacity_pct.eq(10)][['partition','approach','selected','events','detected','recall']].to_string(index=False))
    print('\nSaved:', OUT.relative_to(ROOT))
    print('READ-ONLY. Post-test analysis; not a new independent validation.')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print('STEP 19 FAILED:',exc,file=sys.stderr)
        sys.exit(1)
