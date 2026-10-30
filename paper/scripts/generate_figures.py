#!/usr/bin/env python3
"""Reproduce manuscript figures from saved numerical reports, never rounded prose.

Run from repository root: uv run --no-sync python paper/scripts/generate_figures.py
The manifest records SHA256 hashes and JSON paths for every plotted record.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vers-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_config import apply_paper_style, COLORBLIND_COLORS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper/figures'
RECORDS = []
SOURCES = {}
BLUE, RED, GREEN, PURPLE = COLORBLIND_COLORS[:4]


def read(path):
    raw = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def record(figure, source, pointer, values):
    RECORDS.append(dict(figure=figure, source=source, json_path=pointer, values=values))


def save(fig, name):
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT / f'{name}.{suffix}')
    plt.close(fig)


def point(ax, y, value, ci=None, color=BLUE, marker='o'):
    err = None if ci is None else np.array([[value-ci[0]], [ci[1]-value]])
    ax.errorbar(value, y, xerr=err, color=color, marker=marker, linestyle='none',
                capsize=2.5, markersize=5, elinewidth=1.25)


def decorate(ax, labels, xlabel, margin=False):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.axvline(0, color='#777777', lw=.8)
    ax.grid(axis='y', visible=False)
    if margin:
        ax.axvline(-2, color='#777777', ls='--', lw=1)


def baseline_panel(data, path, models, name):
    fig, axes = plt.subplots(len(models), 2, figsize=(10.2, 3.7 * len(models)),
                             layout='constrained', squeeze=False)
    for row, model in enumerate(models):
        key = f'{model} confirm400'
        n = 'N@a1.0d256' if model == '1.7B' else 'NU26@a0.75d256'
        tuned = 'SEAL@a1.5' if model == '1.7B' else 'SEAL@a2.0'
        n_label = r'VERS, $\alpha=1$' if model == '1.7B' else r'VERS, $\alpha=0.75$'
        arms = [(n, n_label), ('ban_para_d256','Matched word ban'),
                ('ban_all','Global word ban'), ('SEAL','SEAL, default'),
                (tuned, 'SEAL, tuned'), ('DEER','DEER')]
        if model == '1.7B': arms.insert(1, ('N@a0.5d256', r'VERS, $\alpha=0.5$'))
        else: arms.insert(1, ('DU26@a0.5d256', 'Mapped source D'))
        for y, (arm, label) in enumerate(arms):
            r = data['sets'][key]['arms'][arm]
            cost = 'saving_with_trial' if arm == 'DEER' else 'saving'
            color = BLUE if 'VERS' in label else (GREEN if arm == 'DEER' else '#666666')
            point(axes[row,0], y, r[cost], r[cost+'_ci'], color)
            point(axes[row,1], y, r['dacc'], r['dacc_ci'], color)
            record(name,path,['sets',key,'arms',arm],
                   dict(saving=r[cost], saving_ci=r[cost+'_ci'], dacc=r['dacc'],
                        dacc_ci=r['dacc_ci'], cost_field=cost))
        labels = [x[1] for x in arms]
        decorate(axes[row,0],labels,'Generated tokens saved (%)')
        decorate(axes[row,1],labels,'Accuracy change (percentage points)',True)
        axes[row,0].set_title(f'Qwen3-{model}: cost')
        axes[row,1].set_title(f'Qwen3-{model}: accuracy')
        axes[row,0].set_xlim(0,46)
        axes[row,1].set_xlim(-8,2)
    save(fig,name)


PARETO_ARMS = [
    ('N@a1.0d256', r'VERS, $\alpha=1$', BLUE, 'o', (32.45, -4.3)),
    ('N@a0.5d256', r'VERS, $\alpha=0.5$', BLUE, 'o', (30.3, -1.3)),
    ('ban_para_d256', 'Matched word ban', RED, 's', (22.0, -4.3)),
    ('ban_all', 'Global word ban', RED, 's', (31.0, -6.0)),
    ('SEAL', 'SEAL, default', PURPLE, '^', (10.5, -0.6)),
    ('SEAL@a1.5', 'SEAL, tuned', PURPLE, '^', (19.2, -1.4)),
    ('DEER', 'DEER', GREEN, 'D', (38.1, -2.6)),
]


def baseline_pareto(data, path, name):
    key = '1.7B confirm400'
    fig, ax = plt.subplots(figsize=(5.6, 4.0), layout='constrained')
    ax.axhline(0, color='#777777', lw=.8)
    ax.axhline(-2, color='#777777', ls='--', lw=1)
    box = dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=.85)
    ax.text(44.5, -1.92, r'$-2$ pp tolerance', ha='right', va='bottom', fontsize=9,
            color='#555555', bbox=box)
    for arm, label, color, marker, where in PARETO_ARMS:
        r = data['sets'][key]['arms'][arm]
        cost = 'saving_with_trial' if arm == 'DEER' else 'saving'
        x, xci, y, yci = r[cost], r[cost + '_ci'], r['dacc'], r['dacc_ci']
        ax.errorbar(x, y, xerr=[[x - xci[0]], [xci[1] - x]], yerr=[[y - yci[0]], [yci[1] - y]],
                    color=color, marker=marker, linestyle='none', capsize=2.5,
                    markersize=6, elinewidth=1.1, markeredgecolor='white')
        ax.text(*where, label, ha='left', va='center', fontsize=8.5, color=color, bbox=box)
        record(name, path, ['sets', key, 'arms', arm],
               dict(saving=x, saving_ci=xci, dacc=y, dacc_ci=yci, cost_field=cost))
    ax.set_xlim(10, 45)
    ax.set_ylim(-7.5, 1)
    ax.set_xlabel('Generated tokens saved (%)')
    ax.set_ylabel('Accuracy change (pp)')
    ax.text(44.5, 0.75, 'better: right and up', ha='right', va='top', fontsize=9,
            color='#555555', style='italic')
    save(fig, name)


def baseline_figures():
    path = 'data/baselines/report.json'
    data = read(path)
    baseline_panel(data, path, ['1.7B', '8B'], 'baseline_comparison')
    baseline_panel(data, path, ['1.7B'], 'baseline_comparison_1_7b')
    baseline_pareto(data, path, 'baseline_pareto_1_7b')

    fig, axes = plt.subplots(1,2,figsize=(9.4,3.5),layout='constrained')
    models = [('1.7B','N@a1.0d256'),('8B','NU26@a0.75d256'),
              ('30B','NU34@a0.75d256'),('235B','NU67@a1.5d256')]
    for i,(model,arm) in enumerate(models):
        key=f'{model} hard128'
        r=data['sets'][key]['arms'][arm]
        point(axes[0],i,r['saving'],r['saving_ci'])
        point(axes[1],i,r['dacc'],r['dacc_ci'], RED)
        record('competition_scale',path,['sets',key,'arms',arm],r)
    labels=['1.7B','8B','30B-A3B','235B-A22B']
    decorate(axes[0],labels,'Generated tokens saved (%)')
    decorate(axes[1],labels,'Accuracy change (percentage points)',True)
    axes[0].set_title('VERS on hard128: cost')
    axes[1].set_title('VERS on hard128: accuracy')
    axes[0].set_xlim(0,30)
    axes[1].set_xlim(-11,1)
    save(fig,'competition_scale')


def fidelity():
    path='data/xfer_claim4/t1/report_mix.json'
    data=read(path)
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
    for ax,model,title in zip(axes,['8b','30b'],['Qwen3-8B','Qwen3-30B-A3B']):
        d=data[model+'_mix']['web+chat']
        mp=f'data/xfer_claim4/ceiling_{model}.json'
        mlp=read(mp)['affine+MLP hidden 4096']['T1 eval FVE (map ceiling)']['web+chat']
        keys=['T  U(AR(AV(E h_b))) vs h_b','S->U U(AR(AV(h_s))) vs h_b',
              'M  U(h_s) vs h_b','MLP','C  U(E h_b) vs h_b']
        labels=['Adapted NLA (T)','Mapped source readout','Ridge from source state',
                'MLP from source state','No-text target cycle']
        for i,key in enumerate(keys):
            value=mlp if key=='MLP' else d[key]
            point(ax,i,value,color=BLUE if i==0 else '#777777')
            ax.annotate(f'{value:.3f}',(value,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
            record('transfer_fidelity',mp if key=='MLP' else path,
                   ['affine+MLP hidden 4096','T1 eval FVE (map ceiling)','web+chat'] if key=='MLP'
                   else [model+'_mix','web+chat',key],dict(fve=value))
        decorate(ax,labels,'Target-space FVE')
        ax.set_xlim(0,.82)
        ax.set_title(title)
    save(fig,'transfer_fidelity')


def distillation():
    path='data/distill_8b/report.json'
    data=read(path)['readings']['4_diagnostic']
    fig,axes=plt.subplots(2,2,figsize=(10.2,6.2),layout='constrained')
    entries=[('sft_short_lora_vs_base','Natural Short-SFT'),
             ('sft_steered_lora_vs_base','Steered-SFT'),
             ('dpo_natural_lora_vs_base','Natural DPO'),
             ('dpo_steered_lora_vs_base','Steered DPO'),
             ('online_teacher_vs_online_base','Teacher with hook')]
    for row,(cohort,title) in enumerate([('mathfresh','MATH fresh1000'),('mathgate','MATH gate400 (L4–5)')]):
        for y,(key,label) in enumerate(entries):
            r=data[cohort][key]
            color=BLUE if 'Steered' in label else (GREEN if 'Teacher' in label else '#777777')
            marker='s' if 'Teacher' in label else 'o'
            point(axes[row,0],y,-r['dtokens_pct'],[-r['dtokens_ci'][1],-r['dtokens_ci'][0]],color,marker)
            point(axes[row,1],y,r['dacc_pp'],r['dacc_ci'],color,marker)
            record('distillation',path,['readings','4_diagnostic',cohort,key],r)
        labels=[x[1] for x in entries]
        decorate(axes[row,0],labels,'Generated tokens saved (%)')
        decorate(axes[row,1],labels,'Accuracy change (percentage points)',True)
        axes[row,0].set_title(title+': cost')
        axes[row,1].set_title(title+': accuracy')
        axes[row,0].set_xlim(-5,40)
        axes[row,1].set_xlim(-3,1)
    save(fig,'distillation')


def reverse_edit():
    path='data/vers/select.json'
    data=read(path)
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.5),layout='constrained')
    labels=[]
    for i,alpha in enumerate(['0.25','0.5','1.0']):
        key=f'P@a{alpha}d256';r=data[key];labels.append(r'$\alpha='+alpha+'$')
        point(axes[0],i,100*r['length'],color=RED)
        point(axes[1],i,r['ddoubt'],r['ddoubt_ci'],RED)
        record('reverse_edit',path,[key],r)
    decorate(axes[0],labels,'Generated token increase (%)')
    decorate(axes[1],labels,'Change in doubt paragraphs per rollout')
    axes[0].set_title('Fixed doubt-edit vector: longer traces')
    axes[1].set_title('No established increase in doubt paragraphs')
    axes[0].set_xlim(-.5,17)
    save(fig,'reverse_edit')


def alpha_sweep():
    path='paper/figures/source_data/policy_alpha_regraded.json'
    if not (ROOT/path).exists():
        return
    data=read(path)
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.9),layout='constrained')
    for level,color,offset in [('L4',BLUE,-.008),('L5',RED,.008)]:
        for alpha in [.25,.5,.75,1.0]:
            key=f'N@a{alpha}d256'; r=data[level][key]
            for ax,metric,lo,hi in [(axes[0],'save_pct','s_lo','s_hi'),(axes[1],'d_pp','d_lo','d_hi')]:
                ax.errorbar(alpha+offset,r[metric],yerr=[[r[metric]-r[lo]],[r[hi]-r[metric]]],
                            marker='o',ls='none',color=color,capsize=3,label=level if alpha==.25 else None)
            record('strength_difficulty',path,[level,key],r)
    for ax in axes:
        ax.set_xticks([.25,.5,.75,1])
        ax.set_xlabel(r'Steering strength $\alpha$')
        ax.legend()
        ax.axhline(0,color='#777777',lw=.8)
    axes[0].set_ylabel('Generated tokens saved (%)')
    axes[1].set_ylabel('Accuracy change (percentage points)')
    axes[0].set_title('MATH-500: cost by difficulty')
    axes[1].set_title('MATH-500: accuracy by difficulty')
    axes[1].axhline(-2,color='#777777',ls='--',lw=1)
    save(fig,'strength_difficulty')


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    apply_paper_style()
    baseline_figures()
    fidelity()
    distillation()
    reverse_edit()
    alpha_sweep()
    (OUT/'source_data/plotted_records.json').write_text(json.dumps(RECORDS,indent=2)+'\n')
    (OUT/'source_data/manifest.json').write_text(json.dumps({
        'sources_sha256':SOURCES,
        'notes':['Pointwise intervals are copied from source reports, not newly inferred.',
                 'DEER cost includes trial-answer generation.',
                 'Fidelity and reverse-edit token increase have no plotted confidence intervals.',
                 'Dashed accuracy threshold is the stated minus 2 pp non-inferiority margin.',
                 'One training seed for distillation; its intervals quantify evaluation sampling only.']
    },indent=2)+'\n')
    print('Generated:', ', '.join(sorted({r['figure'] for r in RECORDS})))

if __name__=='__main__':main()
