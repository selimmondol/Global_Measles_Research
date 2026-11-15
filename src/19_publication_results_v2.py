#!/usr/bin/env python3
"""Step 19 v2: journal-style visual evidence package, no retraining or tuning.
Run from Measles project: python src/19_publication_results_v2.py
Requires the original Steps 15–18 outputs (same inputs as Step 19 v1),
plus results/validation_design/split_assignments.csv for Fig. 1 if present.
Creates six figures in PNG (450 dpi), PDF, and SVG; tables and provenance.
Everything after opening Step 15 test data is POST-TEST descriptive.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
S15=ROOT/'results'/'independent_test_v1'
S16=ROOT/'results'/'independent_test_reliability_v1'
S17=ROOT/'results'/'outcome_robustness_posttest_v1'
S18=ROOT/'results'/'surveillance_sensitivity_posttest_v1'
OUT=ROOT/'results'/'publication_results_v2'
FIG=OUT/'figures'; TAB=OUT/'tables'
METHODS=['coverage_shortfall_rank','vaccination_only','combined','rf_regularized','xgb_regularized']
PROBS=METHODS[1:]
LABEL={'coverage_shortfall_rank':'Coverage ranking','vaccination_only':'Vaccination LR',
       'combined':'Combined LR','rf_regularized':'Random Forest','xgb_regularized':'XGBoost'}
COLOR={'coverage_shortfall_rank':'#A97855','vaccination_only':'#AAB3BD',
       'combined':'#64748B','rf_regularized':'#173B63','xgb_regularized':'#1B8A87'}
PARTS=['test','unseen_country_test']
PART_LABEL={'test':'Future-year test · 2022–2025','unseen_country_test':'Unseen-country test · 2022–2025'}


def read(folder,name,cols):
    path=folder/name
    if not path.exists(): raise FileNotFoundError(f'Missing {path}. Run earlier research step before Step 19 v2.')
    df=pd.read_csv(path)
    absent=set(cols)-set(df.columns)
    if absent: raise ValueError(f'{path}: missing columns {sorted(absent)}')
    return df

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.titlesize':12,
      'axes.labelsize':10,'axes.edgecolor':'#C5CDD5','axes.labelcolor':'#25364A',
      'xtick.color':'#35465B','ytick.color':'#35465B','text.color':'#152C46',
      'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
      'axes.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,
      'svg.fonttype':'none','axes.titleweight':'bold'})

def polish(ax,axis='y'):
    ax.grid(axis=axis,color='#E6EBEF',linewidth=.75,zorder=0)
    ax.set_axisbelow(True)

def export(fig,stem):
    fig.savefig(FIG/f'{stem}.png',dpi=450,bbox_inches='tight',pad_inches=.14)
    fig.savefig(FIG/f'{stem}.pdf',bbox_inches='tight',pad_inches=.14)
    fig.savefig(FIG/f'{stem}.svg',bbox_inches='tight',pad_inches=.14)
    plt.close(fig)

def s(df,filename): df.to_csv(TAB/filename,index=False,float_format='%.9g')

def getrow(df,**conditions):
    a=df
    for k,v in conditions.items(): a=a[a[k].eq(v)]
    if len(a)!=1: raise ValueError(f'Expected one row for {conditions}; got {len(a)}')
    return a.iloc[0]

def check(pred,cls,pooled,receipt):
    if int(receipt.get('frozen_manifest_files_checked',-1))!=28:
        raise ValueError('Step 15 receipt does not confirm 28 frozen hashes')
    if pred.duplicated(['partition','Country','Year']).any(): raise ValueError('Duplicate test country-year')
    if set(pred.partition.unique())!=set(PARTS): raise ValueError('Unexpected test partition')
    for p,n,e in [('test',505,143),('unseen_country_test',119,33)]:
        d=pred[pred.partition.eq(p)]
        if len(d)!=n or int(d.y.sum())!=e: raise ValueError(f'Unexpected {p}: rows/events')
        for m in PROBS:
            if d[m].isna().any() or not d[m].between(0,1).all(): raise ValueError(f'Invalid probability {m}')
            a=getrow(cls,partition=p,model=m)
            actual=[roc_auc_score(d.y,d[m]),average_precision_score(d.y,d[m]),brier_score_loss(d.y,d[m])]
            recorded=[a.AUROC,a.average_precision,a.brier_score]
            if not np.allclose(actual,recorded,atol=.00016): raise ValueError(f'Metric mismatch {p}/{m}')
        for m in METHODS:
            a=getrow(pooled,partition=p,approach=m,capacity_pct=10)
            if int(a.events)!=e: raise ValueError('Pooled event count mismatch')


def fig1_flow():
    # Grounded on the frozen script's checked partition counts, not invented record exclusions.
    fig,ax=plt.subplots(figsize=(12.5,5.2));ax.set_xlim(0,13);ax.set_ylim(0,6);ax.axis('off')
    boxes=[(0.35,4.28,3.2,1.1,'WHO case reports + WUENIC\nWorld Bank population','2000–2025 · 5,642 country-year rows'),
           (4.85,4.28,3.3,1.1,'Outcome-eligible observations','Definition C · 3,764 rows'),
           (9.25,4.28,3.25,1.1,'Locked research design','40 countries held out'),
           (0.25,1.35,2.65,1.15,'Train 2003–2018','2,144 eligible rows'),
           (3.32,1.35,2.65,1.15,'Validation 2019–2021','396 eligible rows'),
           (6.4,1.35,2.65,1.15,'Future test 2022–2025','505 eligible rows'),
           (9.5,1.35,3.1,1.15,'Unseen-country test','119 eligible rows')]
    for i,(x,y,w,h,title,sub) in enumerate(boxes):
        fc='#F0F6F9' if i<3 else ('#E5F3F0' if i>4 else '#EBF0F7')
        edge='#A5C6C9' if i>4 else '#A9BBCB'
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.13,rounding_size=.12',facecolor=fc,edgecolor=edge,linewidth=1.1))
        ax.text(x+w/2,y+h*.65,title,ha='center',va='center',fontsize=10,fontweight='bold')
        ax.text(x+w/2,y+h*.25,sub,ha='center',va='center',fontsize=8.4,color='#486078')
    for x1,x2 in [(3.63,4.66),(8.23,9.08)]:
        ax.add_patch(FancyArrowPatch((x1,4.82),(x2,4.82),arrowstyle='-|>',mutation_scale=15,color='#56718B',lw=1.3))
    ax.plot([10.9,10.9],[4.13,3.42],color='#56718B',lw=1.2)
    ax.plot([1.58,11.12],[3.42,3.42],color='#56718B',lw=1.2)
    for x in [1.58,4.64,7.74,11.0]:
        ax.add_patch(FancyArrowPatch((x,3.42),(x,2.64),arrowstyle='-|>',mutation_scale=13,color='#56718B',lw=1.1))
    ax.text(6.5,.75,'Excluded / not used in model evaluation: 1,878 outcome-ineligible rows and 600 reserved held-out-country historical rows',
            ha='center',fontsize=8.5,color='#546A7B')
    ax.set_title('Figure 1  |  Study data and independent evaluation design',loc='left',pad=13)
    export(fig,'fig01_study_design')


def fig2_performance(cls):
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.6),sharey=True,layout='constrained')
    for ax,p in zip(axes,PARTS):
        d=cls[cls.partition.eq(p)].set_index('model').loc[PROBS]
        y=np.arange(len(PROBS));offset=.13
        for field,lab,mark in [('AUROC','AUROC','o'),('average_precision','Average precision','s')]:
            vals=d[field].to_numpy(float)
            ys=y+(-offset if field=='AUROC' else offset)
            ax.scatter(vals,ys,s=75 if field=='AUROC' else 55,marker=mark,color=[COLOR[m] for m in PROBS],
                       edgecolor='white',linewidth=.7,zorder=3,label=lab)
            for val,yy in zip(vals,ys):ax.text(val+.018,yy,f'{val:.3f}',va='center',fontsize=8)
        ax.set_yticks(y,[LABEL[m] for m in PROBS]);ax.invert_yaxis()
        ax.set_xlim(.25,1.01);ax.set_xlabel('Discrimination score (higher is better)')
        ax.set_title(PART_LABEL[p],loc='left');polish(ax,'x')
    axes[0].legend(loc='lower right',frameon=False,ncol=2)
    fig.suptitle('Figure 2  |  Held-out model discrimination',x=.02,ha='left',fontweight='bold',fontsize=13)
    export(fig,'fig02_discrimination')


def fig3_capacity(pooled):
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True,layout='constrained')
    chosen=['coverage_shortfall_rank','combined','rf_regularized','xgb_regularized']
    for ax,p in zip(axes,PARTS):
        x=np.arange(len(chosen)); width=.36
        for i,cap in enumerate([10,20]):
            values=[100*float(getrow(pooled,partition=p,approach=m,capacity_pct=cap).recall) for m in chosen]
            bars=ax.bar(x+(i-.5)*width,values,width,color=['#D5DFE9','#7892A9'][i],edgecolor='white',zorder=3,label=f'Top {cap}% capacity')
            for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+.8,f'{v:.1f}',ha='center',fontsize=7.8)
        ax.set_xticks(x,[LABEL[m] for m in chosen],rotation=15,ha='right')
        ax.set_ylim(0,60);ax.set_title(PART_LABEL[p],loc='left');polish(ax)
        ax.text(.02,.96,'Ranking uses outcome-evaluable countries',transform=ax.transAxes,va='top',fontsize=8,color='#687C90')
    axes[0].set_ylabel('Pooled annual recall (%)');axes[0].legend(frameon=False,loc='upper left',bbox_to_anchor=(0,0.88))
    fig.suptitle('Figure 3  |  Annual risk prioritisation at fixed selection capacities',x=.02,ha='left',fontsize=13,fontweight='bold')
    export(fig,'fig03_decision_capacity')


def cal_bins(pred):
    rows=[];summary=[]
    for p in PARTS:
        d=pred[pred.partition.eq(p)]
        for m in PROBS:
            prob=d[m].to_numpy(float); yy=d.y.to_numpy(int)
            b=np.minimum((prob*10).astype(int),9)
            for j in range(10):
                idx=(b==j)
                rows.append(dict(partition=p,model=m,bin=j,n=int(idx.sum()),mean_predicted=float(prob[idx].mean()) if idx.any() else np.nan,
                                 observed_rate=float(yy[idx].mean()) if idx.any() else np.nan))
            summary.append(dict(partition=p,model=m,n=len(d),events=int(yy.sum()),mean_predicted=float(prob.mean()),
                          observed_rate=float(yy.mean()),brier_score=float(brier_score_loss(yy,prob))))
    return pd.DataFrame(rows),pd.DataFrame(summary)


def fig4_calibration(bins,pred):
    # Calibration and bin support are deliberately paired: small bins are not visually over-interpreted.
    fig=plt.figure(figsize=(12.6,8),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[3,1.12])
    for col,p in enumerate(PARTS):
        ax=fig.add_subplot(gs[0,col]); bot=fig.add_subplot(gs[1,col],sharex=ax)
        ax.plot([0,1],[0,1],'--',color='#96A1AB',lw=1,label='Perfect calibration')
        for m in PROBS:
            d=bins[(bins.partition==p)&(bins.model==m)&(bins.n>0)].sort_values('bin')
            ax.plot(d.mean_predicted,d.observed_rate,'o-',lw=1.5,markersize=4.4,color=COLOR[m],label=LABEL[m])
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_ylabel('Observed event frequency' if col==0 else '')
        ax.set_title(PART_LABEL[p],loc='left');polish(ax)
        d=bins[(bins.partition==p)&(bins.model=='rf_regularized')]
        bot.bar((d['bin']+.5)/10,d.n,width=.089,color=COLOR['rf_regularized'],alpha=.82)
        bot.set_xlabel('Predicted risk');bot.set_ylabel('RF bin count' if col==0 else '');polish(bot)
        ax.legend(frameon=False,loc='upper left',fontsize=8)
    fig.suptitle('Figure 4  |  Reliability curves and bin support (10 equal-width bins)',x=.015,ha='left',fontweight='bold',fontsize=13)
    export(fig,'fig04_calibration_and_support')


def fig5_generalization(annual,pooled):
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    ax=axes[0]
    for m in ['coverage_shortfall_rank','rf_regularized','xgb_regularized']:
        d=annual[(annual.partition=='test')&(annual.capacity_pct==10)&(annual.approach==m)].sort_values('Year')
        ax.plot(d.Year,100*d.recall,marker='o',lw=2,markersize=5,label=LABEL[m],color=COLOR[m])
    ax.set_xticks([2022,2023,2024,2025]);ax.set_ylim(0,80);ax.set_ylabel('Annual Recall@10% (%)')
    ax.set_title('A  |  Temporal variation, future test',loc='left');polish(ax);ax.legend(frameon=False,fontsize=8)
    ax=axes[1]; chosen=['coverage_shortfall_rank','combined','rf_regularized','xgb_regularized'];x=np.arange(len(chosen))
    a=[100*getrow(pooled,partition='test',approach=m,capacity_pct=10).recall for m in chosen]
    b=[100*getrow(pooled,partition='unseen_country_test',approach=m,capacity_pct=10).recall for m in chosen]
    ax.scatter(a,x-.11,s=90,c=[COLOR[m] for m in chosen],marker='o',label='Future-year test',zorder=4)
    ax.scatter(b,x+.11,s=75,facecolors='none',edgecolors=[COLOR[m] for m in chosen],marker='D',linewidths=1.8,label='Unseen-country test',zorder=4)
    for j in range(len(x)):ax.plot([a[j],b[j]],[x[j]-.11,x[j]+.11],color='#BBC6D0',lw=1.4,zorder=2)
    ax.set_yticks(x,[LABEL[m] for m in chosen]);ax.invert_yaxis();ax.set_xlim(0,55)
    ax.set_xlabel('Pooled annual Recall@10% (%)');ax.set_title('B  |  Geographic generalisation',loc='left');polish(ax,'x')
    ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Figure 5  |  Temporal stability and held-out-country performance',x=.015,ha='left',fontweight='bold',fontsize=13)
    export(fig,'fig05_temporal_geographic')


def fig6_robustness(rob):
    chosen=['C_primary_frozen','A_broad_2x_1pm','B_strict_3x_1pm','C_exclude_zero_median',
            'C_threshold_2pm','C_threshold_10pm','strict_new_C_onset']
    pretty={'C_primary_frozen':'Primary Definition C','A_broad_2x_1pm':'Broad Definition A',
           'B_strict_3x_1pm':'Stricter Definition B','C_exclude_zero_median':'Exclude zero baseline',
           'C_threshold_2pm':'Threshold ≥2/million','C_threshold_10pm':'Threshold ≥10/million',
           'strict_new_C_onset':'Strict NEW onset'}
    fig,axes=plt.subplots(1,2,figsize=(13.3,6.4),sharey=True,layout='constrained')
    for ax,p in zip(axes,PARTS):
        y=np.arange(len(chosen));w=.24
        for j,m in enumerate(['coverage_shortfall_rank','rf_regularized','xgb_regularized']):
            values=[]
            for o in chosen:
                row=getrow(rob,partition=p,outcome=o,approach=m,capacity_pct=10)
                values.append(100*float(row.recall))
            ax.barh(y+(j-1)*w,values,height=w,color=COLOR[m],label=LABEL[m],zorder=3)
        ax.set_yticks(y,[pretty[o] for o in chosen]);ax.set_xlim(0,65)
        ax.set_title(PART_LABEL[p],loc='left');ax.set_xlabel('Annual Recall@10% (%)');polish(ax,'x')
        ax.axhline(5.5,color='#B8C3CC',lw=1,linestyle='--')
        ax.text(.97,.05,'Exploratory; altered cohorts',transform=ax.transAxes,ha='right',fontsize=8,color='#64748B')
    axes[0].invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=9,loc='lower center',ncol=3,bbox_to_anchor=(0.5,-0.035))
    fig.suptitle('Figure 6  |  Outcome sensitivity, including new-onset failure',x=.01,ha='left',fontweight='bold',fontsize=13)
    export(fig,'fig06_outcome_robustness')


def main():
    pred=read(S15,'test_predictions.csv',['partition','Country','Year','y']+PROBS)
    cls=read(S15,'test_classification_metrics.csv',['partition','model','AUROC','average_precision','brier_score'])
    annual=read(S15,'annual_decision_metrics.csv',['partition','Year','capacity_pct','approach','recall','events','detected'])
    pooled=read(S15,'pooled_decision_summary.csv',['partition','approach','capacity_pct','events','selected','detected','recall'])
    rob=read(S17,'decision_metrics_by_outcome.csv',['outcome','partition','approach','capacity_pct','recall'])
    sub=read(S18,'subgroup_sensitivity_summary.csv',['partition','grouping','subgroup','n','events'])
    ci=read(S16,'paired_bootstrap_differences.csv',['partition','model_a','observed_difference','lower_95pct','upper_95pct'])
    receiptpath=S15/'evaluation_receipt.json';receipt=json.loads(receiptpath.read_text())
    check(pred,cls,pooled,receipt)
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
    style()
    bins,calsum=cal_bins(pred)
    s(cls,'table01_model_scores.csv');s(pooled,'table02_decision_capacity.csv')
    s(annual,'table03_annual_decisions.csv');s(ci,'table04_bootstrap_intervals.csv')
    s(rob,'table05_outcome_robustness.csv');s(sub,'table06_surveillance_subgroups.csv')
    s(bins,'table07_calibration_bins.csv');s(calsum,'table08_calibration_summary.csv')
    fig1_flow();fig2_performance(cls);fig3_capacity(pooled);fig4_calibration(bins,pred)
    fig5_generalization(annual,pooled);fig6_robustness(rob)
    inputs=[S15/'test_predictions.csv',S15/'test_classification_metrics.csv',S15/'annual_decision_metrics.csv',
            S15/'pooled_decision_summary.csv',S17/'decision_metrics_by_outcome.csv',
            S18/'subgroup_sensitivity_summary.csv',S16/'paired_bootstrap_differences.csv',receiptpath]
    lines=['Step 19 v2 — read-only visual package','Frozen test cohort: 505 future + 119 unseen-country rows.',
       'No training, tuning, recalibration, threshold selection or protocol modification.',
       'Figures use researcher-defined REPORTED-INCIDENCE SURGES; not WHO outbreak confirmation.',
       'Figures are POST-TEST descriptions, not independent replication.','Figure 1 counts from the locked Step 14/15 design and preceding data processing.',
       'Figure 2 reports point estimates only; no uncomputed AUROC/AP confidence intervals are implied.',
       'Figure 3 uses annual top-10% and top-20% selection on outcome-evaluable rows.',
       'Figure 4 uses pooled country-years, 10 equal-width bins; small bins are unstable.',
       'Figure 5 unseen countries share the same future calendar shocks; not a separate historical period.',
       'Figure 6 alternative outcomes and some eligibility cohorts differ; post-test exploratory.',
       'Strict NEW onset results are negative and deliberately shown.','Historical WUENIC revisions and incomplete surveillance prevent a prospective claim.',
       'Input SHA256:']
    lines += [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}' for p in inputs]
    (OUT/'FIGURE_NOTES_AND_PROVENANCE.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('STEP 19 V2: SIX PUBLICATION-STYLE FIGURES COMPLETE')
    print('Freeze receipt checks:',receipt['frozen_manifest_files_checked'],'| test observations:',len(pred))
    print('Figures: 6 PNG + 6 PDF + 6 SVG | Tables: 8 CSV')
    print('Output:',OUT.relative_to(ROOT))
    print('No retraining, tuning, test reselection, or outcome edits.')

if __name__=='__main__':
    try:main()
    except Exception as ex:
        print('STEP 19 V2 FAILED:',str(ex),file=sys.stderr);sys.exit(1)
