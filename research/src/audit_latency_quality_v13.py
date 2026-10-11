"""Post-hoc descriptive quality audit of completed B1 timing, never fitting."""
import signal
from inference_ablations_v13 import *
from score_bank import grade,timeout_handler
from scipy.stats import binomtest

def main():
    run=RUN/'latency_b1';done=json.loads((run/'COMPLETE.json').read_text())
    assert digest(run/'raw.jsonl')==done['raw_sha256']
    raw=readl(run/'raw.jsonl');assert len(raw)==192
    inputs={x['id']:x for x in readl(ROOT/'data/inference_ablations_v13/dev_inputs.jsonl')}
    gold={x['id']:x['answer'] for x in readl(ROOT/'data/inference_ablations_v13/dev_targets.jsonl')}
    signal.signal(signal.SIGALRM,timeout_handler);scored=[]
    for r in raw:
        pred,y,error=grade({**inputs[r['id']],'answer':gold[r['id']]},r['text']);assert error is None
        scored.append({k:r[k] for k in ['id','method','ratio','input_tokens','output_tokens','total_request_seconds']}|{'prediction':pred,'correct':y})
    ids=sorted({r['id'] for r in raw});assert len(ids)==64
    table={(r['id'],r['method']):r for r in scored};rt={(r['id'],r['method']):r for r in raw}
    methods=['fixed_r1','adaptive_relative','matched_mixture_seed17'];ix=np.random.default_rng(20261011).integers(64,size=(10000,64))
    summary={};paired={};a=np.array([table[q,'adaptive_relative']['correct'] for q in ids])
    for m in methods:
        rows=[table[q,m] for q in ids];y=np.array([r['correct'] for r in rows])
        summary[m]={'n':64,'correct':int(y.sum()),'accuracy':float(y.mean()),
         'accuracy_ci95':np.quantile(y[ix].mean(1),[.025,.975]).tolist(),
         'mean_output_tokens':float(np.mean([r['output_tokens'] for r in rows])),
         'retention_counts':{str(r):sum(v['ratio']==r for v in rows) for r in RATIOS}}
        if m!='adaptive_relative':
            w=int(((a==1)&(y==0)).sum());l=int(((a==0)&(y==1)).sum())
            paired[m]={'accuracy_delta':float((a-y).mean()),'accuracy_delta_ci95':np.quantile((a-y)[ix].mean(1),[.025,.975]).tolist(),
             'wins':w,'losses':l,'mcnemar_p_unadjusted_descriptive':float(binomtest(w,w+l).pvalue) if w+l else 1.}
    bank={r['id']:r for r in readl(RUN/'raw.jsonl') if r['experiment_split']=='dev' and r['condition']=='adaptive_relative'}
    route_changes=[q for q in ids if rt[q,'adaptive_relative']['ratio']!=bank[q]['ratio']]
    checks=[]
    for q in ids:
        for m in ['fixed_r1','matched_mixture_seed17']:
            aa=rt[q,'adaptive_relative'];b=rt[q,m]
            if aa['ratio']==b['ratio']:checks.append({'id':q,'reference':m,'ratio':aa['ratio'],'same_tokens':aa['token_ids']==b['token_ids']})
    writel(run/'quality_scores.jsonl',scored)
    result={'scope':'Post-hoc descriptive audit of timing DEV subset, not final screening or policy selection',
      'summary':summary,'paired_accuracy':paired,'same_query_same_ratio_checks':checks,
      'same_ratio_mismatches':sum(not x['same_tokens'] for x in checks),
      'B1_route_changes_from_cached_B16_feature_policy':route_changes,
      'raw_sha256':digest(run/'raw.jsonl'),'code_sha256':digest(__file__),
      'new_training':False,'policy_changed':False,'cost':'No additional model inference; CPU grading of saved timing outputs only.'}
    dump(run/'QUALITY_AUDIT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
