"""Measured paired rainclouds and recovery matrices; no invented values."""
import argparse,json
from pathlib import Path
import numpy as np
from render_v13_scientific_figures import readj,readl,style,frame,finish,C
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    l=readj(a.run/'latency_b1/COMPLETE.json');r=readj(a.run/'recovery_k3/COMPLETE.json');q=readj(a.run/'latency_b1/QUALITY_AUDIT.json')
    raw=readl(a.run/'latency_b1/raw.jsonl');ids=sorted({x['id'] for x in raw});lookup={(x['id'],x['method']):x for x in raw}
    a.output.mkdir(parents=True,exist_ok=True);style();fig=frame('Measured latency and same-CoT recovery diagnostics',
      'Two separate 64-question dev subsets; complete frozen studies with all observations retained')
    gs=fig.add_gridspec(2,2,left=.15,right=.95,top=.82,bottom=.19,wspace=.35,hspace=.5)
    ax,ay,az,aw=[fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    methods=['fixed_r1','adaptive_relative','matched_mixture_seed17'];labels=['Full retention','Adaptive','Budget mixture'];colors=[C['gray'],C['teal'],C['orange']]
    values=np.array([[lookup[i,m]['total_request_seconds'] for m in methods] for i in ids]);jitter=np.random.default_rng(17).uniform(.06,.20,len(ids))
    for n,row in enumerate(values):ax.plot(np.arange(3)+jitter[n],row,c=C['gray'],alpha=.07,lw=.6,zorder=1)
    for i,col in enumerate(colors):
        v=values[:,i];body=ax.violinplot([v],positions=[i],widths=.7,showextrema=False)['bodies'][0]
        xy=body.get_paths()[0].vertices;xy[:,0]=np.minimum(xy[:,0],i);body.set_facecolor(col);body.set_edgecolor(col);body.set_alpha(.4)
        ax.scatter(i+jitter,v,s=13,c=col,alpha=.65,edgecolors='none');lo,med,hi=np.quantile(v,[.25,.5,.75])
        ax.plot([i,i],[lo,hi],c=col,lw=5);ax.scatter(i,med,c='white',edgecolors=col,s=35,zorder=4)
    ax.set_xticks(range(3),labels);ax.set(ylabel='Actual B1 request time (seconds)',title='a  Paired latency rainclouds',xlim=(-.6,2.5))
    rates=[.5,.7,1.];conds=['question_only','true_cot','mismatched_cot']
    mat=np.array([[r['per_ratio_controls'][str(rt)]['mean_recovery_accuracy'][c]*100 for c in conds] for rt in rates])
    im=ay.imshow(mat,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto');ay.grid(False)
    for i in range(3):
        for j in range(3):ay.text(j,i,f'{mat[i,j]:.1f}%',ha='center',va='center',c='white' if mat[i,j]>65 else C['ink'],fontsize=13)
    ay.set_xticks(range(3),['Question only','True CoT','Mismatched CoT']);ay.set_yticks(range(3),['r = .5','r = .7','r = 1.0']);ay.set_title('b  K3 recovery accuracy matrix')
    for i,m in enumerate(['fixed_r1','matched_mixture_seed17']):
        s=l['paired'][m];v=s['mean_latency_delta_seconds'];lo,hi=s['paired_ci95']
        az.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',c=C['teal'],capsize=4,lw=1.8)
    az.axvline(0,ls='--',c=C['gray'],lw=1);az.set_yticks([0,1],['vs full retention','vs budget mixture']);az.invert_yaxis()
    az.set(xlabel='Adaptive minus reference (seconds), 95% paired CI',title='c  End-to-end time difference')
    for i,rt in enumerate(rates):
        for j,(key,color) in enumerate([('true_minus_question_only',C['teal']),('true_minus_mismatched',C['orange'])]):
            s=r['per_ratio_controls'][str(rt)][key];v=s['delta']*100;lo,hi=np.array(s['paired_question_ci95'])*100
            aw.errorbar(v,i+(j-.5)*.22,xerr=[[v-lo],[hi-v]],fmt='o',c=color,capsize=3,lw=1.5,label=['True - question','True - mismatched'][j] if i==0 else None)
    aw.axvline(0,ls='--',c=C['gray'],lw=1);aw.set_yticks(range(3),['r = .5','r = .7','r = 1.0']);aw.invert_yaxis();aw.legend(fontsize=9,loc='best')
    aw.set(xlabel='Paired recovery difference (percentage points)',title='d  CoT controls, question-bootstrap CI')
    acc=', '.join(str(q['summary'][m]['correct']) for m in methods)+' correct out of 64'
    registry=[];finish(fig,a.output,'05_latency_and_recovery',
      f'B1 quality (full/adaptive/mixture): {acc}. Encoder/router included; load/warmup separate. K3: cap32, 325/1728 truncated; no faithfulness proof.',
      ['latency_b1/COMPLETE.json','latency_b1/raw.jsonl','latency_b1/QUALITY_AUDIT.json','recovery_k3/COMPLETE.json'],registry)
    (a.output/'ADVANCED_SUPPLEMENT_MANIFEST.json').write_text(json.dumps({'status':'pending_visual_review','figures':registry},indent=2));print(str(a.output))
if __name__=='__main__':main()
