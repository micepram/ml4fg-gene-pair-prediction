"""Generate original, vector report figures exclusively from saved results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
BLUE='#247BA0';ORANGE='#DC7F35';GREEN='#39856B';GRAY='#737B84';RED='#AD404B'


def save(fig,name):
    fig.savefig(ROOT/f'figures/{name}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/f'figures/{name}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def workflow():
    fig,ax=plt.subplots(figsize=(7.2,.74));ax.set(xlim=(0,10),ylim=(0,1));ax.axis('off')
    boxes=[(.05,'Norman cells\ncondition means'),(2.55,'Pair partitions\ntrain / val / test'),(5.05,'GO + coexpression\nGEARS ± GenePT'),(7.55,'Held-out pairs\nvs additivity')]
    for x,text in boxes:
        ax.add_patch(FancyBboxPatch((x,.2),2.15,.62,boxstyle='round,pad=0.04',linewidth=1,edgecolor=BLUE,facecolor='#EEF5F8'))
        ax.text(x+1.075,.51,text,ha='center',va='center',fontsize=9)
    for x in [2.22,4.72,7.22]:ax.add_patch(FancyArrowPatch((x,.51),(x+.26,.51),arrowstyle='-|>',mutation_scale=12,color=GRAY))
    save(fig,'01_workflow')


def eda():
    d=np.load(ROOT/'data/processed/condition_profiles.npz');meta=pd.read_csv(ROOT/'data/processed/conditions.csv')
    summary=json.loads((ROOT/'data/processed/data_summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.25),layout='constrained')
    ax=axes[0]
    bins=np.geomspace(40,8000,20)
    for typ,col in [('single',BLUE),('pair',ORANGE)]:ax.hist(meta.query('type==@typ').n_cells,bins=bins,alpha=.65,color=col,label=typ)
    ax.axvline(summary['control_cells'],color=GREEN,lw=1.2,ls='--',label='control')
    ax.set(xscale='log',xlabel='Cells per condition',ylabel='Number of conditions',title='A  Uneven cell coverage');ax.legend(frameon=False,fontsize=7)
    ax=axes[1]
    ax.hist(d['expression_sample'],bins=45,color=BLUE,density=True)
    ax.text(.96,.94,f"{summary['zero_fraction']:.1%} zeros",transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax.set(xlabel='Nonzero log expression',ylabel='Density',title='B  Expression distribution')
    ax=axes[2]
    for typ,col,marker,size in [('single',BLUE,'o',11),('pair',ORANGE,'o',11),('control',GREEN,'*',70)]:
        sub=meta.query('type==@typ');ax.scatter(sub.PC1,sub.PC2,s=size,c=col,marker=marker,alpha=.7,label=typ)
    ax.set(xlabel=f"PC1 ({summary['pca_variance_ratio'][0]:.1%})",ylabel=f"PC2 ({summary['pca_variance_ratio'][1]:.1%})",title='C  Condition-mean PCA')
    ax.legend(frameon=False,fontsize=7,loc='best')
    save(fig,'02_eda')
    pca=PCA(n_components=2).fit(d['means'][:,d['top1000']])
    rows=[]
    for k in range(2):
        for j in np.argsort(-np.abs(pca.components_[k]))[:10]:rows.append(dict(component=k+1,gene=d['genes'][d['top1000'][j]],loading=pca.components_[k,j]))
    pd.DataFrame(rows).to_csv(ROOT/'results/pca_top_loadings.csv',index=False)


def performance():
    baseline=pd.read_csv(ROOT/'results/baseline_pair_metrics.csv')
    summary=pd.read_csv(ROOT/'results/baseline_summary.csv')
    pilot=[]
    for slug in ['gears','gears_genept']:
        p=ROOT/f'results/{slug}_pair_metrics.csv'
        if not p.exists():print('Pilot pending:',p.name);return
        pilot.append(pd.read_csv(p))
    common=pd.concat([baseline.query('split==101')]+pilot,ignore_index=True)
    order=['Additive','Training mean','No change','GEARS pilot','GEARS + GenePT pilot']
    labels=['Additive','Train mean','No change','GEARS pilot','+ GenePT pilot']
    colors=[GREEN,BLUE,GRAY,ORANGE,RED]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.05),layout='constrained')
    ax=axes[0]
    for i,method in enumerate(order[:3]):
        row=summary.set_index('method').loc[method]
        ax.bar(i,row.mean_mse,color=colors[i],width=.6,alpha=.85)
        vals=baseline.query('method==@method').groupby('split').mse.mean()
        ax.scatter(np.linspace(i-.14,i+.14,len(vals)),vals,color='black',s=9,zorder=3)
    ax.set(xticks=range(3),xticklabels=labels[:3],ylabel='MSE (top 1,000 control genes)',title='A  Simple baselines: five pair splits')
    ax.tick_params(axis='x',labelsize=8)
    ax.text(.02,.98,'Bars: 98 unique pairs\nDots: split means',transform=ax.transAxes,ha='left',va='top',fontsize=7)
    ax=axes[1];out=[]
    for i,(method,label,color) in enumerate(zip(order,labels,colors)):
        x=common.query('method==@method').paired_improvement.to_numpy()
        rng=np.random.default_rng(1879)
        boot=rng.choice(x,(10000,len(x)),replace=True).mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975]);mean=x.mean()
        ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color=color,capsize=3,ms=5)
        out.append(dict(method=method,n_pairs=len(x),mean_mse=common.query('method==@method').mse.mean(),mean_difference=mean,ci_low=lo,ci_high=hi))
    ax.axvline(0,color='black',lw=.8,ls='--');ax.set(yticks=range(5),yticklabels=labels,xlabel='Additive MSE − model MSE',title='B  Same 31 test pairs: split 101')
    ax.invert_yaxis();ax.tick_params(axis='y',labelsize=8);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    ax.set_xticks([-.03,-.02,-.01,0]);ax.set_xlim(-.033,.002)
    save(fig,'03_performance')
    pd.DataFrame(out).to_csv(ROOT/'results/common_split_comparison.csv',index=False)


if __name__=='__main__':
    workflow();eda();performance()
