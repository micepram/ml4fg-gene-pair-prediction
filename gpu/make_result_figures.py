"""Original figures from verified GPU results; preserves the earlier pilots."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'gpu/results/bounded_3h'
FIGURES = ROOT / 'gpu/figures'
BLUE, ORANGE, GREEN, GRAY = '#247BA0', '#DC7F35', '#39856B', '#737B84'
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(FIGURES / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(7.2, .74))
    ax.set(xlim=(0,10), ylim=(0,1))
    ax.axis('off')
    labels = ['Norman cells\n84,143 retained', 'Pair partitions\nsingles observed',
              'GEARS / extension\nindividual cells', 'Held-out pairs\nvs additivity']
    for x, label in zip([.05,2.55,5.05,7.55], labels):
        ax.add_patch(FancyBboxPatch((x,.2),2.15,.62,boxstyle='round,pad=0.04',
                                   linewidth=1,edgecolor=BLUE,facecolor='#EEF5F8'))
        ax.text(x+1.075,.51,label,ha='center',va='center',fontsize=8.5)
    for x in [2.22,4.72,7.22]:
        ax.add_patch(FancyArrowPatch((x,.51),(x+.26,.51),arrowstyle='-|>',mutation_scale=12,color=GRAY))
    save(fig, '01_workflow')


def performance():
    baseline = pd.read_csv(ROOT / 'interim/results/baseline_summary.csv')
    base_pairs = pd.read_csv(ROOT / 'interim/results/baseline_pair_metrics.csv')
    summary = pd.read_csv(RESULTS / 'summary.csv').query("subset=='top1000'")
    fig, axes = plt.subplots(1,2,figsize=(7.2,2.05),layout='constrained',gridspec_kw={'width_ratios':[1,1.18]})
    ax = axes[0]
    for i,(name,label,color) in enumerate([('Additive','Additive',GREEN),('Training mean','Train mean',BLUE),('No change','No change',GRAY)]):
        value = baseline.set_index('method').loc[name,'mean_mse']
        ax.bar(i,value,color=color,width=.6,alpha=.85)
        points = base_pairs.query('method==@name').groupby('split').mse.mean()
        ax.scatter(np.linspace(i-.14,i+.14,len(points)),points,color='black',s=9,zorder=3)
    ax.set(xticks=range(3),xticklabels=['Additive','Train mean','No change'],ylabel='MSE (top 1,000 control genes)',
           title='A  Simple baselines: five splits')
    ax.tick_params(axis='x',labelsize=8)
    ax.text(.02,.98,'Bars: 98 unique pairs\nDots: split means',transform=ax.transAxes,ha='left',va='top',fontsize=7)
    ax = axes[1]
    for i,(prior,label,color) in enumerate([('native','GEARS',BLUE),('depmap','+ DepMap / attention',ORANGE)]):
        row = summary.set_index('prior').loc[prior]
        error = np.array([[row.paired_improvement-row.ci_low],[row.ci_high-row.paired_improvement]])
        ax.errorbar(row.paired_improvement,i,xerr=error,fmt='o',color=color,capsize=3,ms=5)
    ax.axvline(0,color='black',lw=.8,ls='--')
    ax.set(yticks=[0,1],yticklabels=['GEARS','+ DepMap /\nattention'],
           xlabel='Additive MSE - model MSE',title='B  Cell-level models: split 101',xlim=(-.014,.001))
    ax.invert_yaxis()
    ax.tick_params(axis='y',labelsize=8)
    ax.set_xticks([-.012,-.008,-.004,0])
    ax.text(.02,.53,'31 pairs; five epochs; one seed',transform=ax.transAxes,ha='left',va='top',fontsize=7)
    save(fig, '03_performance')


def training():
    fig, axes = plt.subplots(1,2,figsize=(8,2.7),layout='constrained')
    for name,label,color in [('native_sum','GEARS',BLUE),('depmap_attention','+ DepMap / attention',ORANGE)]:
        h = pd.read_csv(RESULTS / f'bounded_{name}_split101_seed17/history.csv')
        axes[0].plot(h.epoch,h.training_gears_loss,'o-',label=label,color=color)
        axes[1].plot(h.epoch,h.validation_mse,'o-',label=label,color=color)
    axes[0].set(xlabel='Epoch',ylabel='GEARS training loss',title='Fourth-power + direction objective')
    axes[1].set(xlabel='Epoch',ylabel='Validation MSE (log scale)',yscale='log',title='Validation-selected epoch: 3 for both')
    for ax in axes:
        ax.set_xticks(range(1,6))
        ax.legend(frameon=False,fontsize=8)
    save(fig,'04_training')


if __name__ == '__main__':
    workflow()
    performance()
    training()
