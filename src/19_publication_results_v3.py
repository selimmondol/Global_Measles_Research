#!/usr/bin/env python3
"""Step 19 v3: journal-style visual evidence package, no retraining or tuning.
Run from Measles project: python src/19_publication_results_v3.py
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
OUT=ROOT/'results'/'publication_results_v3'
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
    if not path.exists(): raise FileNotFoundError(f'Missing {path}. Run earlier research step before Step 19 v3.')
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
    # Counts are a mutually exclusive accounting of the 3,764 outcome-eligible rows.
    fig,ax=plt.subplots(figsize=(13.2,6.7))
    ax.set_xlim(0,13.2);ax.set_ylim(0,7);ax.axis('off')
    def box(x,y,w,h,title,detail,fc='#EEF3F7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12,rounding_size=.12',
                      facecolor=fc,edgecolor='#9CB2C5',lw=1.35))
        ax.text(x+w/2,y+h*.66,title,ha='center',va='center',weight='bold',fontsize=10.2)
        ax.text(x+w/2,y+h*.25,detail,ha='center',va='center',fontsize=9,color='#4E667A')
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.4,color='#5F7890'))
    box(.45,5.5,4,1.05,'WHO cases · WUENIC · World Bank','2000–2025  |  5,642 country-year rows')
    box(5.1,5.5,3.15,1.05,'Definition C screening','3,764 eligible · 1,878 ineligible')
    box(9.05,5.5,3.6,1.05,'Excluded after screening','1,878 ineligible rows',fc='#F4F1EF')
    arrow((4.6,6.02),(4.95,6.02));arrow((8.37,6.02),(8.90,6.02))
    box(4.65,3.88,3.8,.85,'Locked partition assignments','3,764 eligible rows')
    arrow((6.7,5.35),(6.7,4.91))
    ax.plot([1.55,11.75],[3.46,3.46],color='#5F7890',lw=1.3)
    arrow((6.55,3.75),(6.55,3.46))
    boxes=[(.35,2.06,2.35,'Train 2003–2018','2,144 rows','#E9F0F7'),
           (3.0,2.06,2.35,'Validation 2019–2021','396 rows','#E9F0F7'),
           (5.65,2.06,2.35,'Future test 2022–2025','505 rows','#E1F1EE'),
           (8.30,2.06,2.35,'Unseen countries','119 test rows','#E1F1EE'),
           (10.95,2.06,1.9,'Reserved history','600 rows','#F3F0EA')]
    for x,y,w,t,detail,c in boxes:
        box(x,y,w,.95,t,detail,c);arrow((x+w/2,3.45),(x+w/2,3.14))
    ax.text(6.6,1.23,'Held-out countries: 40. Their historical 600 rows were not used in training or testing.',
            ha='center',fontsize=9.8,color='#40556B')
    ax.text(6.6,.74,'Accounting check: 2,144 + 396 + 505 + 119 + 600 = 3,764 eligible rows',
            ha='center',fontsize=9.2,color='#5C7387')
    ax.set_title('Figure 1  |  Study data and locked evaluation design',loc='left',pad=12,fontsize=14)
    export(fig,'fig01_study_design')

def fig2_performance(cls):
    fig,axes=plt.subplots(1,2,figsize=(13.1,4.8),sharey=True)
    for ax,p in zip(axes,PARTS):
        d=cls[cls.partition.eq(p)].set_index('model').loc[PROBS]
        y=np.arange(len(PROBS))
        for field,lab,mark,shift in [('AUROC','AUROC','o',-.14),('average_precision','Average precision','s',.14)]:
            vals=d[field].to_numpy(float)
            ax.scatter(vals,y+shift,s=85 if mark=='o' else 65,marker=mark,
                color=[COLOR[m] for m in PROBS],edgecolor='white',lw=.7,zorder=4,label=lab)
            for v,yy in zip(vals,y+shift):ax.text(v+.013,yy,f'{v:.3f}',fontsize=8.5,va='center')
        ax.set_yticks(y,[LABEL[m] for m in PROBS]);ax.invert_yaxis()
        ax.set_xlim(.23,1.02);ax.set_xlabel('Score (higher is better)')
        ax.set_title(PART_LABEL[p],loc='left',pad=12);polish(ax,'x')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,.01),fontsize=10)
    fig.subplots_adjust(left=.13,right=.98,top=.83,bottom=.22,wspace=.10)
    fig.suptitle('Figure 2  |  Held-out model discrimination',x=.04,y=.97,ha='left',weight='bold',fontsize=14)
    export(fig,'fig02_discrimination')

def fig3_capacity(pooled,ci):
    fig,axes=plt.subplots(1,2,figsize=(13.0,5.5),sharey=True)
    chosen=['coverage_shortfall_rank','combined','rf_regularized','xgb_regularized']
    for ax,p in zip(axes,PARTS):
        x=np.arange(len(chosen));w=.35
        for j,cap in enumerate([10,20]):
            vals=[100*getrow(pooled,partition=p,approach=m,capacity_pct=cap).recall for m in chosen]
            bars=ax.bar(x+(j-.5)*w,vals,w,color=['#CEDBE5','#738FA8'][j],edgecolor='white',zorder=3,
                        label=f'Top {cap}% capacity')
            for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.75,f'{v:.1f}',ha='center',fontsize=8)
        ax.set_xticks(x,[LABEL[m] for m in chosen],rotation=12,ha='right');ax.set_ylim(0,60)
        ax.set_title(PART_LABEL[p],loc='left');polish(ax)
        ax.text(.02,.98,'Outcome-evaluable country-years only',va='top',transform=ax.transAxes,fontsize=8.1,color='#637A8C')
    axes[0].set_ylabel('Pooled annual recall (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.012))
    # CI is from paired country-cluster bootstrap of Recall@10% vs coverage ranking, NOT bar-wise CIs.
    for ax,p in zip(axes,PARTS):
        d=ci[(ci.partition==p)&(ci.model_a=='rf_regularized')]
        if len(d)==1:
            r=d.iloc[0]
            note=f"RF − coverage at 10%: {100*r.observed_difference:+.1f} pp (95% CI {100*r.lower_95pct:+.1f}, {100*r.upper_95pct:+.1f})"
            ax.text(.50,-.32,note,ha='center',va='top',transform=ax.transAxes,fontsize=8.5,color='#364F69')
    fig.subplots_adjust(top=.84,bottom=.29,left=.08,right=.98,wspace=.11)
    fig.suptitle('Figure 3  |  Annual risk prioritisation at fixed capacities',x=.03,y=.97,ha='left',fontsize=14,weight='bold')
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
    # Quantile bins are defined separately per model/partition, solely to VISUALISE frozen predictions.
    # This is post-test descriptive aggregation, not recalibration.
    from sklearn.calibration import calibration_curve
    models=['rf_regularized','xgb_regularized']
    fig,axes=plt.subplots(2,2,figsize=(12.4,8.5),sharex='col',gridspec_kw={'height_ratios':[3.1,1.25]})
    for col,p in enumerate(PARTS):
        d=pred[pred.partition==p]; y=d.y.to_numpy(int)
        ax=axes[0,col];bot=axes[1,col]
        ax.plot([0,1],[0,1],ls='--',lw=1.2,color='#9AA9B6',label='Ideal calibration',zorder=1)
        for m in models:
            v=d[m].to_numpy(float)
            # Identical scores can reduce number of occupied quantile bins.
            q=pd.qcut(v,q=5,labels=False,duplicates='drop')
            tab=pd.DataFrame({'risk':v,'outcome':y,'bin':q}).groupby('bin',observed=True).agg(
                mean_risk=('risk','mean'),observed=('outcome','mean'),n=('outcome','size')).reset_index()
            ax.plot(tab.mean_risk,tab.observed,'o-',color=COLOR[m],lw=1.8,ms=6,label=LABEL[m])
            for r in tab.itertuples(index=False):
                if r.n<15:ax.annotate(f'n={r.n}',(r.mean_risk,r.observed),xytext=(4,5),textcoords='offset points',fontsize=7.5)
            if m=='rf_regularized':
                bot.bar(tab.mean_risk,tab.n,width=.07,alpha=.65,color=COLOR[m],edgecolor='white',label='RF bin count')
        ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_title(PART_LABEL[p],loc='left',pad=12)
        ax.set_ylabel('Observed surge frequency' if col==0 else '')
        bot.set_ylabel('RF bin n' if col==0 else '');bot.set_xlabel('Mean predicted risk per bin')
        polish(ax);polish(bot)
        ax.legend(frameon=False,loc='upper left',fontsize=9)
        bot.text(.98,.94,'RF quantile bins (up to 5)',transform=bot.transAxes,ha='right',va='top',fontsize=8,color='#526A7C')
    fig.suptitle('Figure 4  |  Calibration: ML models and bin support',x=.04,ha='left',y=.98,fontsize=14,weight='bold')
    fig.text(.5,.016,'Post-test descriptive curves; small bins and country-year dependence limit precision. No recalibration performed.',
             ha='center',fontsize=8.8,color='#526A7C')
    fig.subplots_adjust(top=.87,bottom=.09,left=.08,right=.98,hspace=.17,wspace=.14)
    export(fig,'fig04_calibration_and_support')

def fig5_generalization(annual,pooled):
    fig,axes=plt.subplots(1,2,figsize=(13.1,5.25))
    ax=axes[0]
    for m in ['coverage_shortfall_rank','rf_regularized','xgb_regularized']:
        d=annual[(annual.partition=='test')&(annual.capacity_pct==10)&(annual.approach==m)].sort_values('Year')
        style={'xgb_regularized':dict(ls='--',marker='s',markersize=5,alpha=.78),
               'rf_regularized':dict(ls='-',marker='o',markersize=6),
               'coverage_shortfall_rank':dict(ls='-',marker='^',markersize=5)}[m]
        ax.plot(d.Year,100*d.recall,lw=2,color=COLOR[m],label=LABEL[m],**style)
    src=annual[(annual.partition=='test')&(annual.capacity_pct==10)&(annual.approach=='rf_regularized')]
    for r in src.itertuples():
        ax.text(r.Year,69,f'{int(r.events)} events',ha='center',fontsize=8,color='#586F82')
    ax.set_xticks([2022,2023,2024,2025]);ax.set_ylim(0,78)
    ax.set_ylabel('Annual Recall@10% (%)');ax.set_title('A  |  Future-test variation',loc='left')
    polish(ax);ax.legend(frameon=False,fontsize=8,loc='upper right')
    ax=axes[1];chosen=['coverage_shortfall_rank','combined','rf_regularized','xgb_regularized'];yy=np.arange(len(chosen))
    a=[100*getrow(pooled,partition='test',approach=m,capacity_pct=10).recall for m in chosen]
    b=[100*getrow(pooled,partition='unseen_country_test',approach=m,capacity_pct=10).recall for m in chosen]
    for j,m in enumerate(chosen):ax.plot([a[j],b[j]],[yy[j]-.12,yy[j]+.12],color='#B8C5D0',lw=1.6)
    ax.scatter(a,yy-.12,s=93,c=[COLOR[m] for m in chosen],label='Future-year test',zorder=4)
    ax.scatter(b,yy+.12,s=83,facecolors='none',edgecolors=[COLOR[m] for m in chosen],marker='D',lw=1.8,
               label='Unseen-country test',zorder=4)
    ax.set_yticks(yy,[LABEL[m] for m in chosen]);ax.invert_yaxis();ax.set_xlim(0,51)
    ax.set_xlabel('Pooled annual Recall@10% (%)');ax.set_title('B  |  Held-out-country performance',loc='left')
    polish(ax,'x');ax.legend(frameon=False,fontsize=8)
    fig.subplots_adjust(top=.83,bottom=.15,left=.08,right=.98,wspace=.34)
    fig.suptitle('Figure 5  |  Temporal and geographic generalisation',x=.03,y=.96,ha='left',weight='bold',fontsize=14)
    export(fig,'fig05_temporal_geographic')

def fig6_robustness(rob,cohorts):
    chosen=['C_primary_frozen','A_broad_2x_1pm','B_strict_3x_1pm','C_exclude_zero_median',
            'C_threshold_2pm','C_threshold_10pm','strict_new_C_onset']
    pretty={'C_primary_frozen':'Primary Definition C','A_broad_2x_1pm':'Broad Definition A',
            'B_strict_3x_1pm':'Stricter Definition B','C_exclude_zero_median':'Exclude zero baseline',
            'C_threshold_2pm':'Threshold ≥2/million','C_threshold_10pm':'Threshold ≥10/million',
            'strict_new_C_onset':'Strict NEW onset'}
    fig,axes=plt.subplots(1,2,figsize=(14.1,7.0),sharey=True)
    for ax,p in zip(axes,PARTS):
        y=np.arange(len(chosen));w=.24
        for j,m in enumerate(['coverage_shortfall_rank','rf_regularized','xgb_regularized']):
            vals=[100*getrow(rob,partition=p,outcome=o,approach=m,capacity_pct=10).recall for o in chosen]
            ax.barh(y+(j-1)*w,vals,height=w,color=COLOR[m],zorder=3,label=LABEL[m])
        for pos,o in enumerate(chosen):
            d=getrow(cohorts,partition=p,outcome=o)
            ax.text(63,pos,f'n={int(d.evaluated)}; events={int(d.events)}',va='center',fontsize=8.0,color='#52697E')
        ax.set_xlim(0,98);ax.set_xticks([0,10,20,30,40,50,60])
        ax.set_yticks(y,[pretty[o] for o in chosen]);ax.set_title(PART_LABEL[p],loc='left')
        ax.set_xlabel('Annual Recall@10% (%)');polish(ax,'x')
        ax.axhline(5.5,color='#B5C3CE',lw=1,ls='--')
        ax.text(.63,.045,'Post-test exploratory',transform=ax.transAxes,ha='left',fontsize=8.5,color='#5B7185')
    axes[0].invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.02))
    fig.subplots_adjust(left=.16,right=.98,top=.87,bottom=.13,wspace=.12)
    fig.suptitle('Figure 6  |  Outcome sensitivity and new-onset limitation',x=.02,y=.97,ha='left',weight='bold',fontsize=14)
    export(fig,'fig06_outcome_robustness')

def main():
    pred=read(S15,'test_predictions.csv',['partition','Country','Year','y']+PROBS)
    cls=read(S15,'test_classification_metrics.csv',['partition','model','AUROC','average_precision','brier_score'])
    annual=read(S15,'annual_decision_metrics.csv',['partition','Year','capacity_pct','approach','recall','events','detected'])
    pooled=read(S15,'pooled_decision_summary.csv',['partition','approach','capacity_pct','events','selected','detected','recall'])
    rob=read(S17,'decision_metrics_by_outcome.csv',['outcome','partition','approach','capacity_pct','recall'])
    cohorts=read(S17,'outcome_cohort_summary.csv',['outcome','partition','evaluated','events'])
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
    fig1_flow();fig2_performance(cls);fig3_capacity(pooled,ci);fig4_calibration(bins,pred)
    fig5_generalization(annual,pooled);fig6_robustness(rob,cohorts)
    inputs=[S15/'test_predictions.csv',S15/'test_classification_metrics.csv',S15/'annual_decision_metrics.csv',
            S15/'pooled_decision_summary.csv',S17/'decision_metrics_by_outcome.csv',S17/'outcome_cohort_summary.csv',
            S18/'subgroup_sensitivity_summary.csv',S16/'paired_bootstrap_differences.csv',receiptpath]
    lines=['Step 19 v3 — read-only visual package','Frozen test cohort: 505 future + 119 unseen-country rows.',
       'No training, tuning, recalibration, threshold selection or protocol modification.',
       'Figures use researcher-defined REPORTED-INCIDENCE SURGES; not WHO outbreak confirmation.',
       'Figures are POST-TEST descriptions, not independent replication.','Figure 1 counts from the locked Step 14/15 design and preceding data processing.',
       'Figure 2 reports point estimates only; no uncomputed AUROC/AP confidence intervals are implied.'
       ,'Figure 3 reports paired country-cluster bootstrap 95% intervals for RF minus coverage ranking only, not intervals for all bars.',
       'Figure 3 uses annual top-10% and top-20% selection on outcome-evaluable rows.',
       'Figure 4 uses up to 5 within-model quantile bins for descriptive visualization; the frozen probability predictions are unchanged. Bin counts are shown for RF.',
       'Figure 5 unseen countries share the same future calendar shocks; not a separate historical period.',
       'Figure 6 shows the original cohort sizes and event counts for each alternative outcome; changed cohorts mean recall values are not directly matched.',
       'Strict NEW onset results are negative and deliberately shown.','Historical WUENIC revisions and incomplete surveillance prevent a prospective claim.',
       'Input SHA256:']
    lines += [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}' for p in inputs]
    (OUT/'FIGURE_NOTES_AND_PROVENANCE.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('STEP 19 V3: SIX PUBLICATION-STYLE FIGURES COMPLETE')
    print('Freeze receipt checks:',receipt['frozen_manifest_files_checked'],'| test observations:',len(pred))
    print('Figures: 6 PNG + 6 PDF + 6 SVG | Tables: 8 CSV')
    print('Output:',OUT.relative_to(ROOT))
    print('No retraining, tuning, test reselection, or outcome edits.')

if __name__=='__main__':
    try:main()
    except Exception as ex:
        print('STEP 19 V2 FAILED:',str(ex),file=sys.stderr);sys.exit(1)
