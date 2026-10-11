"""Render exact V13 evidence after analysis, never invent incomplete results.

Usage: python render_v13_scientific_figures.py --run EXTRACTED_RUN --output OUT
PNG exports are native 3840 x 2160; vector PDF/SVG and source tables accompany them.
This is a reporting utility, not an experiment or policy selection program.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

C = {'ink':'#18344A', 'teal':'#008D8A', 'orange':'#D87D31',
     'gray':'#7D8993', 'light':'#E8EDF0', 'blue':'#5B82BA', 'red':'#C25360'}
NAMES = {'adaptive_relative':'Adaptive relative', 'fixed_1.0':'Full retention',
 'best_global_fixed_dev':'Dev-selected fixed', 'matched_mixture_seed17':'Dev-budget mixture',
 'length_heuristic':'Length heuristic', 'p1_difficulty_heuristic':'Difficulty heuristic',
 'without_correctness_gate':'Without quality gate', 'without_learned_cost':'Without cost head',
 'absolute_tau_0p97':'Absolute 0.97 gate', 'unconditional_dev_prediction':'Unconditional route',
 'shuffled_actions_seed17':'Shuffled routes (17)',
 'existing_recipe_soft_cost':'Without boundary + policy block',
 'existing_recipe_soft_mono_boundary':'+ Soft monotonicity',
 'existing_recipe_asymmetric_soft_mono_boundary':'+ Monotonicity + asymmetric BCE'}

def readj(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def readl(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s]

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
      'axes.labelsize':13, 'axes.titlesize':15, 'axes.titleweight':'bold',
      'axes.labelcolor':C['ink'], 'text.color':C['ink'], 'xtick.color':C['ink'],
      'ytick.color':C['ink'], 'axes.edgecolor':'#CBD4DB', 'axes.spines.top':False,
      'axes.spines.right':False, 'axes.grid':True, 'grid.color':C['light'],
      'grid.linewidth':.7, 'axes.axisbelow':True, 'legend.frameon':False,
      'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
      'savefig.facecolor':'white', 'figure.facecolor':'white'})

def frame(title, subtitle):
    fig = plt.figure(figsize=(16,9))
    fig.suptitle(title, x=.07, y=.97, ha='left', fontsize=23, weight='bold')
    fig.text(.07,.918,subtitle,fontsize=11,color=C['gray'])
    return fig

def finish(fig,out,name,foot,sources,registry):
    fig.text(.07,.027,foot,fontsize=10,color=C['gray'])
    files=[]
    for ext in ['png','pdf','svg']:
        dest=out/f'{name}.{ext}'
        fig.savefig(dest,dpi=240)  # no tight crop: exactly 3840 x 2160 pixels
        files.append({'file':dest.name,'sha256':digest(dest)})
    plt.close(fig)
    registry.append({'figure':name,'png_pixels':[3840,2160],
      'renderer':'matplotlib; measured data, not AI-generated coordinates',
      'input_files':sources,'files':files})

def save_table(out,name,rows):
    (out/f'{name}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if rows and isinstance(rows,list) and isinstance(rows[0],dict):
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (out/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def plot_frontier(run,out,m,p,registry):
    fig=frame('Accuracy and actual decoding cost',
       'CSQA reused screening, n = 256 | Qwen2.5-7B + author GSM adapter | matched-width V13')
    ax,ay=fig.subplots(1,2,gridspec_kw={'width_ratios':[1.45,1]})
    fig.subplots_adjust(left=.07,right=.97,bottom=.18,top=.84,wspace=.28)
    fixed=[f'fixed_{r:.1f}' for r in np.arange(3,11)/10]
    x=[m[k]['mean_output_tokens'] for k in fixed];y=[m[k]['accuracy']*100 for k in fixed]
    ax.scatter(x,y,color=C['gray'],s=38,label='Fixed retention (0.3–1.0)')
    # Descriptive fixed-policy frontier only: do not connect dominated points
    # into an apparently monotone retention-versus-cost curve.
    frontier=[];best=-np.inf
    for xx,yy in sorted(zip(x,y)):
        if yy>best:frontier.append((xx,yy));best=yy
    ax.plot(*np.array(frontier).T,color=C['gray'],lw=1,ls=':',label='Observed fixed-policy frontier')
    for k,xx,yy in zip(fixed,x,y):
        ax.annotate(k.split('_')[-1],(xx,yy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
    for k,col,marker in [('adaptive_relative',C['teal'],'*'),
       ('matched_mixture_seed17',C['orange'],'D'),('best_global_fixed_dev',C['blue'],'s')]:
        q=m[k];xx=q['mean_output_tokens'];yy=100*q['accuracy']
        ax.errorbar(xx,yy,xerr=np.array([[xx-q['mean_output_tokens_ci95'][0]],
          [q['mean_output_tokens_ci95'][1]-xx]]),yerr=np.array([[yy-100*q['accuracy_ci95'][0]],
          [100*q['accuracy_ci95'][1]-yy]]),fmt=marker,color=col,ms=13 if marker=='*' else 8,
          lw=1.2,capsize=3,label=NAMES[k],zorder=4)
    ax.set(xlabel='Mean generated tokens per question (including EOS)',ylabel='Accuracy (%)')
    ax.legend(loc='best',fontsize=10)
    keys=['fixed_1.0','best_global_fixed_dev','matched_mixture_seed17','length_heuristic','p1_difficulty_heuristic']
    for i,k in enumerate(keys):
        q=p[k];v=100*q['output_saving_fraction'];lo,hi=np.array(q['output_saving_ci95'])*100
        ay.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=C['teal'],capsize=4,lw=1.8)
    ay.axvline(0,color=C['gray'],lw=1,ls='--');ay.set_yticks(range(len(keys)),[NAMES[k] for k in keys]);ay.invert_yaxis()
    ay.set(xlabel='Adaptive output-token saving vs reference (%)',title='Paired savings, 95% bootstrap CI')
    finish(fig,out,'01_accuracy_cost_frontier',
      'Exploratory screening; mixture selected using dev only. Budget matching is expected on dev, not guaranteed on screen. CIs are question bootstraps.',
      ['METHOD_RESULTS.json','PAIRED_COMPARISONS.json','DEV_BASELINE_SELECTION.json'],registry)

def plot_ablations(run,out,m,p,registry):
    fig=frame('Inference ablations: quality–cost trade-offs',
      'Difference is Adaptive relative minus each reference; all existing checkpoints remain frozen')
    ax,ay=fig.subplots(1,2,sharey=True);fig.subplots_adjust(left=.26,right=.95,top=.83,bottom=.19,wspace=.28)
    keys=['without_correctness_gate','without_learned_cost','absolute_tau_0p97',
       'unconditional_dev_prediction','shuffled_actions_seed17','existing_recipe_soft_cost',
       'existing_recipe_soft_mono_boundary','existing_recipe_asymmetric_soft_mono_boundary']
    rows=[]
    for i,k in enumerate(keys):
        q=p[k];v=q['accuracy_delta']*100;lo,hi=np.array(q['accuracy_delta_ci95'])*100
        col=C['teal'] if lo>0 else C['red'] if hi<0 else C['ink']
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=col,capsize=4,lw=1.5)
        z=q['output_saving_fraction']*100;l,h=np.array(q['output_saving_ci95'])*100
        ay.errorbar(z,i,xerr=[[z-l],[h-z]],fmt='o',color=C['teal'],capsize=4,lw=1.5)
        rows.append({'reference':k,**q})
    for a in [ax,ay]:a.axvline(0,color=C['gray'],ls='--',lw=1)
    ax.set_yticks(range(len(keys)),[NAMES[k] for k in keys]);ax.invert_yaxis()
    ax.set(xlabel='Paired accuracy difference (percentage points)',title='Accuracy, 95% CI')
    ay.set(xlabel='Output-token saving (%)',title='Cost, 95% CI')
    finish(fig,out,'02_inference_ablation_forest',
      'Cached fixed-bank policies; no per-policy latency claimed. Existing recipe comparisons are not newly trained single-component ablations. All signs retained.',
      ['PAIRED_COMPARISONS.json','METHOD_RESULTS.json'],registry)
    save_table(out,'02_inference_ablation_source',rows)

def plot_mechanism(run,out,m,d,registry):
    rows=readl(run/'scores_screen.jsonl');ids=sorted({x['id'] for x in rows});rates=[.5,.6,.7,.8,.9,1.]
    tab={(x['id'],x['condition']):x for x in rows};Y=np.array([[tab[q,f'fixed_{r:.1f}']['correct'] for r in rates] for q in ids])
    # Deterministic grouping is only a display order; every query is shown.
    order=sorted(range(len(ids)),key=lambda i:(tuple(Y[i]),ids[i]));groups=[]
    for r in rates:groups.append(m['adaptive_relative']['retention_counts'].get(str(r),0))
    fig=frame('What the controller compresses — and where the assumption fails',
      'All 256 screening questions shown; ordering by correctness pattern is for display only')
    gs=fig.add_gridspec(2,2,left=.08,right=.95,top=.84,bottom=.17,wspace=.3,hspace=.45,width_ratios=[1.45,1])
    ax=fig.add_subplot(gs[:,0]);ay=fig.add_subplot(gs[0,1]);az=fig.add_subplot(gs[1,1])
    ax.imshow(Y[order].T,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#E9EEF1',C['teal']]),vmin=0,vmax=1)
    ax.set_yticks(range(6),[f'{r:.1f}' for r in rates]);ax.set(xlabel='Question index (all rows, grouped by outcome pattern)',ylabel='Retention ratio',title='Fixed-bank correctness: teal = correct')
    ax.grid(False)
    ay.bar(np.arange(6),groups,color=[C['teal']]*5+[C['gray']],width=.72)
    ay.set_xticks(range(6),[f'{r:.1f}' for r in rates]);ay.set(xlabel='Chosen retention',ylabel='Questions',title='Actual adaptive allocation')
    bins=d['calibration']['bins'];az.plot([0,1],[0,1],ls='--',lw=1,color=C['gray'])
    az.scatter([b['predicted'] for b in bins],[b['observed'] for b in bins],
       s=[30+200*b['n']/sum(z['n'] for z in bins) for b in bins],color=C['orange'],edgecolors='white',lw=.7)
    az.set(xlim=(-.02,1.04),ylim=(-.02,1.04),xlabel='Frozen predicted correctness',ylabel='Observed correctness',title='Transferred-controller calibration')
    note=f"Nonmonotone outcomes (.5–1): {d['nonmonotone_binary_correctness_fraction_primary_range']*100:.1f}% of questions. "
    note+=f"Brier = {d['calibration']['Brier']:.3f}; ECE = {d['calibration']['ECE_equal_width10']:.3f}. No target-domain calibration fitted."
    finish(fig,out,'03_allocation_motivation_calibration',note,['scores_screen.jsonl','DIAGNOSTICS.json','METHOD_RESULTS.json'],registry)
    save_table(out,'03_correctness_matrix_source',[{'id':ids[i],**{f'r{r}':int(Y[i,j]) for j,r in enumerate(rates)}} for i in order])

def plot_seeds(run,out,m,registry):
    fig=frame('Robustness across existing controller seeds and budget-mixture draws',
       'Seeds 17 / 43 / 79 were trained earlier; this study performs inference only')
    ax,ay=fig.subplots(1,2);fig.subplots_adjust(left=.09,right=.96,top=.82,bottom=.19,wspace=.28)
    for prefix,color,label in [('controller_seed',C['teal'],'Independent controller'),
       ('matched_mixture_seed',C['orange'],'Budget-mixture draw'),('shuffled_actions_seed',C['blue'],'Shuffled routing')]:
        for seed in [17,43,79]:
            q=m[f'{prefix}{seed}'];ax.scatter(q['mean_output_tokens'],q['accuracy']*100,c=color,s=75,label=label if seed==17 else None)
            ax.annotate(str(seed),(q['mean_output_tokens'],q['accuracy']*100),xytext=(5,5),textcoords='offset points',fontsize=10)
    ax.scatter(m['adaptive_relative']['mean_output_tokens'],m['adaptive_relative']['accuracy']*100,marker='*',c=C['ink'],s=170,label='Three-controller ensemble')
    ax.set(xlabel='Mean generated tokens',ylabel='Accuracy (%)',title='Seed sensitivity');ax.legend(fontsize=10)
    for i,(k,color,label) in enumerate([('fixed_1.0',C['gray'],'Full retention'),('adaptive_relative',C['teal'],'Adaptive relative'),('matched_mixture_seed17',C['orange'],'Dev-budget mixture')]):
        values=np.array([x['output_tokens'] for x in readl(run/'policy_rows_screen.jsonl') if x['method']==k])
        body=ay.violinplot([values],positions=[i],vert=False,widths=.65,showextrema=False)['bodies'][0]
        vertices=body.get_paths()[0].vertices;vertices[:,1]=np.minimum(vertices[:,1],i)
        body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.4)
        jitter=np.random.default_rng(701+i).uniform(.06,.20,len(values))
        ay.scatter(values,i+jitter,s=7,color=color,alpha=.28,edgecolors='none')
        q=np.quantile(values,[.1,.25,.5,.75,.9]);ay.plot([q[0],q[4]],[i,i],c=color,lw=1.1)
        ay.plot([q[1],q[3]],[i,i],c=color,lw=5);ay.scatter(q[2],i,c='white',edgecolor=color,s=35,zorder=4)
    ay.set_yticks(range(3),['Full retention','Adaptive relative','Dev-budget mixture']);ay.invert_yaxis()
    ay.set(xlabel='Generated tokens per question',title='Output-length rainclouds',ylim=(2.6,-.6))
    finish(fig,out,'04_seed_and_length_robustness',
       'Three ensemble members are not three independent end-to-end experimental repetitions. Mixture seeds vary allocation, not model weights.',
       ['METHOD_RESULTS.json','SEED_STUDY.json','policy_rows_screen.jsonl'],registry)

def plot_supplements(run,out,registry):
    lat=run/'latency_b1'/'COMPLETE.json';rec=run/'recovery_k3'/'COMPLETE.json'
    if not (lat.exists() and rec.exists()):return
    l=readj(lat);r=readj(rec);lr=readl(lat.parent/'raw.jsonl')
    fig=frame('End-to-end latency and same-CoT recovery diagnostics',
       'Separate 64-question dev subsets | B1 timing is measured; recovery is an offline annotation diagnostic')
    ax,ay=fig.subplots(1,2);fig.subplots_adjust(left=.08,right=.97,top=.83,bottom=.2,wspace=.26)
    methods=['fixed_r1','adaptive_relative','matched_mixture_seed17'];short=['Full retention','Adaptive','Budget mixture'];colors=[C['gray'],C['teal'],C['orange']]
    for i,(method,col) in enumerate(zip(methods,colors)):
        values=np.array([x['total_request_seconds'] for x in lr if x['method']==method]);v=np.sort(values)
        ax.plot(v,(np.arange(len(v))+1)/len(v),color=col,lw=2,label=f"{short[i]} (median {np.median(v):.2f}s)")
        ax.axvline(np.quantile(v,.95),color=col,lw=.8,ls=':')
    ax.set(xlabel='Actual request time (seconds)',ylabel='Empirical cumulative fraction',title='Single-request latency, including routing');ax.legend(fontsize=10)
    ratios=[.5,.7,1.];xx=np.arange(3);width=.23
    for j,(cond,col,lab) in enumerate([('question_only',C['gray'],'Question only'),('true_cot',C['teal'],'True CoT'),('mismatched_cot',C['orange'],'Mismatched CoT')]):
        vals=[100*r['per_ratio_controls'][str(v)]['mean_recovery_accuracy'][cond] for v in ratios]
        ay.bar(xx+(j-1)*width,vals,width,color=col,label=lab)
    ay.set_xticks(xx,['0.5','0.7','1.0']);ay.set(xlabel='Source CoT retention',ylabel='Recovery accuracy (%)',title='K = 3, same CoT across independent answers',ylim=(0,100));ay.legend(fontsize=10)
    finish(fig,out,'05_latency_and_recovery',
       'One timing trial per query/method, rotated order; loading/warmup separate. Recovery K3/K5 does not establish a trained soft-label ablation or reasoning faithfulness.',
       ['latency_b1/COMPLETE.json','latency_b1/raw.jsonl','recovery_k3/COMPLETE.json'],registry)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    run=a.run.resolve();out=a.output.resolve();assert run!=out
    assert (run/'ANALYSIS_COMPLETE.json').is_file(),'Bank analysis is incomplete; no draft results fabricated.'
    done=readj(run/'ANALYSIS_COMPLETE.json');assert done['identity_passed'],'Same-r identity mismatch requires review before rendering causal comparisons.'
    assert digest(run/'raw.jsonl')==done['raw_sha256']
    out.mkdir(parents=True,exist_ok=True);style();registry=[]
    m=readj(run/'METHOD_RESULTS.json');p=readj(run/'PAIRED_COMPARISONS.json');d=readj(run/'DIAGNOSTICS.json')
    assert m['adaptive_relative']['n']==256
    plot_frontier(run,out,m,p,registry);plot_ablations(run,out,m,p,registry)
    plot_mechanism(run,out,m,d,registry);plot_seeds(run,out,m,registry);plot_supplements(run,out,registry)
    manifest={'status':'rendered_pending_visual_QA','code_sha256':digest(__file__),
      'analysis_sha256':digest(run/'ANALYSIS_COMPLETE.json'),'figures':registry,
      'figure_type':'exact scientific plots; no image-generation model used for data coordinates',
      'not_a_claim':'These exploratory inference studies do not complete prohibited retraining ablations or establish four-domain superiority.'}
    (out/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'figures':len(registry),'output':str(out),'visual_QA':'required'}))

if __name__=='__main__':main()
