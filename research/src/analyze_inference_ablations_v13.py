"""Inference-only ablation analysis; all baseline fitting uses dev, no model fitting."""
import time,signal,csv
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score,average_precision_score
from inference_ablations_v13 import *
from score_bank import grade,timeout_handler
from transformers import AutoTokenizer

def boot_ci(x,index):return np.quantile(np.asarray(x)[index].mean(1),[.025,.975]).tolist()
def hash_uniform(qid,seed):return int.from_bytes(hashlib.sha256(f'{seed}|{qid}'.encode()).digest()[:8],'big')/2**64
def metrics(rows,tau,index):
    y=np.array([x['correct'] for x in rows]);c=np.array([x['output_tokens'] for x in rows]);pc=np.array([x['paper_cot_tokens'] for x in rows]);n=len(y);acc=float(y.mean());cost=float(c.mean())
    return {'n':n,'correct':int(y.sum()),'accuracy':acc,'accuracy_ci95':boot_ci(y,index),'mean_output_tokens':cost,'mean_output_tokens_ci95':boot_ci(c,index),'mean_paper_cot_tokens':float(pc.mean()),'total_output_tokens':int(c.sum()),'median_output_tokens':float(np.median(c)),'p90_output_tokens':float(np.quantile(c,.9)),'mean_input_tokens':float(np.mean([x['input_tokens'] for x in rows])),'truncated':sum(x['truncated'] for x in rows),'correct_per_1k_tokens':float(1000*y.sum()/c.sum()),'tokens_per_correct':float(c.sum()/y.sum()) if y.sum() else None,'E3':acc**2/cost,'BCTU':{str(beta):float(np.mean(y/(1+(c/tau)**beta))) for beta in [.5,1.,2.]},'retention_counts':{str(r):sum(x['ratio']==r for x in rows) for r in RATIOS}}
def paired(ar,br,index):
    y=np.array([x['correct'] for x in ar]);b=np.array([x['correct'] for x in br]);c=np.array([x['output_tokens'] for x in ar]);d=np.array([x['output_tokens'] for x in br]);wins=int(((y==1)&(b==0)).sum());losses=int(((y==0)&(b==1)).sum())
    saving=1-c[index].mean(1)/d[index].mean(1)
    return {'accuracy_delta':float((y-b).mean()),'accuracy_delta_ci95':boot_ci(y-b,index),'output_token_delta':float((c-d).mean()),'output_token_delta_ci95':boot_ci(c-d,index),'output_saving_fraction':float(1-c.mean()/d.mean()),'output_saving_ci95':np.quantile(saving,[.025,.975]).tolist(),'wins':wins,'losses':losses,'mcnemar_p':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.}
def holm(comps,names):
    ordered=sorted(names,key=lambda k:comps[k]['mcnemar_p']);previous=0
    for i,k in enumerate(ordered):previous=max(previous,min(1.,(len(ordered)-i)*comps[k]['mcnemar_p']));comps[k]['holm_p']=previous

def main():
    lock=(RUN/'analysis.lock').open('a')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (RUN/'ANALYSIS_COMPLETE.json').exists():return
    while not (RUN/'BANK_COMPLETE.json').exists():
        status=json.loads((RUN/'status.json').read_text())
        if status.get('status')=='failed':raise RuntimeError(status)
        dump(RUN/'analysis_status.json',{'status':'waiting_for_bank','completed_generations':status.get('completed_generations')});time.sleep(10)
    f=settings();done=json.loads((RUN/'BANK_COMPLETE.json').read_text());assert done['freeze_sha256']==digest(FREEZE) and digest(RUN/'raw.jsonl')==done['raw_sha256'] and digest(RUN/'batches.jsonl')==done['batches_sha256']
    for p,h in f['data_files'].items():assert digest(ROOT/p)==h
    raw=readl(RUN/'raw.jsonl');assert len(raw)==4608 and all(x['generation_requests']==1 for x in raw)
    tokenizer=AutoTokenizer.from_pretrained(ADAPTER,local_files_only=True);signal.signal(signal.SIGALRM,timeout_handler)
    pred=np.load(ROOT/'data/inference_ablations_v13/predictions.npz');allids=pred['ids'].tolist();pmap={q:i for i,q in enumerate(allids)}
    scored={};tables={};inputs={};indices=np.random.default_rng(20261010).integers(256,size=(10000,256))
    for split in ['dev','screen']:
        # Strict order: DEV scored and reference parameters frozen before SCREEN labels opened.
        dump(RUN/'analysis_status.json',{'status':'scoring','split':split})
        inputs[split]=readl(ROOT/f'data/inference_ablations_v13/{split}_inputs.jsonl');gold={x['id']:x['answer'] for x in readl(ROOT/f'data/inference_ablations_v13/{split}_targets.jsonl')};items={x['id']:{**x,'answer':gold[x['id']]} for x in inputs[split]};rows=[]
        for x in raw:
            if x['experiment_split']!=split:continue
            prediction,y,e=grade(items[x['id']],x['text']);assert e is None,(x['id'],e)
            rows.append({k:x[k] for k in ['id','condition','ratio','output_tokens','output_tokens_excluding_eos','input_tokens','truncated','batch_key']}|{'correct':y,'prediction':prediction,'paper_cot_tokens':len(tokenizer(x['text'].split('\n\nThe final answer is:')[0])['input_ids'])})
        assert len(rows)==256*9;writel(RUN/f'scores_{split}.jsonl',rows);scored[split]=rows;table={(r['id'],r['condition']):r for r in rows};tables[split]=table;ids=[x['id'] for x in inputs[split]]
        if split=='dev':
            cost=np.array([[table[i,f'fixed_{r:.1f}']['output_tokens'] for r in RATIOS] for i in ids]);Y=np.array([[table[i,f'fixed_{r:.1f}']['correct'] for r in RATIOS] for i in ids]);means=cost.mean(0);acc=Y.mean(0);feasible=acc>=acc[-1];best=int(np.argmin(np.where(feasible,means,np.inf)));target=float(np.mean([table[i,'adaptive_relative']['output_tokens'] for i in ids]));options=[]
            for j in range(7):
                if means[j]!=means[j+1] and min(means[j:j+2])<=target<=max(means[j:j+2]):options.append((abs(means[j+1]-means[j]),j,float((target-means[j])/(means[j+1]-means[j]))))
            if options:_,lo,w=sorted(options)[0];hi=lo+1
            else:lo=hi=int(np.argmin(abs(means-target)));w=0.
            pi=[pmap[i] for i in ids];a=actions(pred['prob_soft_boundary'][pi],pred['cost_soft_boundary'][pi]);hist=np.array([np.mean(a==j) for j in range(2,8)]);levels=np.array(RATIOS[2:]);quant=np.cumsum(hist)[:-1]
            lengths=np.array([table[i,'fixed_1.0']['input_tokens'] for i in ids]);difficulty=-pred['prob_soft_boundary'][pi,-1]
            select={'status':'frozen_from_dev_only','main_policy_changed':False,'model_training':False,'screen_labels_opened_for_this_bank':False,'global_fixed_ratio':RATIOS[best],'global_selection_rule':f['analysis_plan']['best_global_fixed'],'matched_mixture':{'low_ratio':RATIOS[lo],'high_ratio':RATIOS[hi],'probability_high':w,'target_dev_actual_output_tokens':target,'expected_dev_mixture_output_tokens':float((1-w)*means[lo]+w*means[hi]),'seeds':list(SEEDS),'screen_cost_matching_not_guaranteed':True},'heuristic_levels':levels.tolist(),'length_cutoffs':np.quantile(lengths,quant).tolist(),'difficulty_cutoffs':np.quantile(difficulty,quant).tolist(),'unconditional_ratio_from_dev_predictions':float(RATIOS[actions(pred['prob_soft_boundary'][pi].mean(0,keepdims=True),pred['cost_soft_boundary'][pi].mean(0,keepdims=True))[0]]),'tau_BCTU':float(means[-1]),'dev_fixed_accuracy':acc.tolist(),'dev_fixed_output_tokens':means.tolist(),'primary_dev_action_histogram':hist.tolist()}
            dump(RUN/'DEV_BASELINE_SELECTION.json',select)
        else:assert (RUN/'DEV_BASELINE_SELECTION.json').exists()
    select=json.loads((RUN/'DEV_BASELINE_SELECTION.json').read_text());split='screen';table=tables[split];ids=[x['id'] for x in inputs[split]];pi=[pmap[i] for i in ids];P=pred['prob_soft_boundary'][pi];C=pred['cost_soft_boundary'][pi];n=256;methods={};modes={}
    def attach(name,act,mode='cached_fixed_bank_counterfactual'):
        methods[name]=[{**table[q,f'fixed_{RATIOS[int(j)]:.1f}'],'method':name,'evaluation_mode':mode} for q,j in zip(ids,act)];modes[name]=mode
    for j,r in enumerate(RATIOS):attach(f'fixed_{r:.1f}',np.full(n,j),'actual_fixed_generation')
    methods['adaptive_relative']=[{**table[q,'adaptive_relative'],'method':'adaptive_relative','evaluation_mode':'actual_single_generation'} for q in ids];modes['adaptive_relative']='actual_single_generation'
    for variant,label in [('absolute','absolute_tau_0p97'),('no_quality','without_correctness_gate'),('no_cost','without_learned_cost')]:attach(label,actions(P,C,variant))
    attach('unconditional_dev_prediction',np.full(n,RATIOS.index(select['unconditional_ratio_from_dev_predictions'])))
    attach('best_global_fixed_dev',np.full(n,RATIOS.index(select['global_fixed_ratio'])))
    mix=select['matched_mixture'];lo=RATIOS.index(mix['low_ratio']);hi=RATIOS.index(mix['high_ratio'])
    primary_actions=actions(P,C)
    for seed in SEEDS:
        attach(f'matched_mixture_seed{seed}',[hi if hash_uniform(q,seed)<mix['probability_high'] else lo for q in ids])
        attach(f'shuffled_actions_seed{seed}',np.random.default_rng(seed).permutation(primary_actions))
        attach(f'controller_seed{seed}',actions(pred[f'prob_seed{seed}'][pi],pred[f'cost_seed{seed}'][pi]))
    levels=np.array(select['heuristic_levels']);lengths=np.array([table[q,'fixed_1.0']['input_tokens'] for q in ids]);difficulty=-P[:,-1]
    for name,values,cut in [('length_heuristic',lengths,select['length_cutoffs']),('p1_difficulty_heuristic',difficulty,select['difficulty_cutoffs'])]:
        ratios=levels[np.searchsorted(cut,values,side='right')];attach(name,[RATIOS.index(float(r)) for r in ratios])
    for recipe in ['soft_cost','soft_mono_boundary','asymmetric_soft_mono_boundary']:attach('existing_recipe_'+recipe,actions(pred['prob_'+recipe][pi],pred['cost_'+recipe][pi]))
    Y=np.array([[table[q,f'fixed_{r:.1f}']['correct'] for r in RATIOS] for q in ids]);TC=np.array([[table[q,f'fixed_{r:.1f}']['output_tokens'] for r in RATIOS] for q in ids]);valid=np.array(RATIOS)>=.5
    oc=np.where((Y==1)&valid[None,:],TC,np.inf);oa=np.argmin(oc,axis=1);none=~((Y==1)&valid[None,:]).any(1);oa[none]=np.argmin(np.where(valid[None,:],TC,np.inf),axis=1)[none];attach('oracle_selected_cost_lower_bound',oa,'label_dependent_oracle_not_deployable')
    result={name:metrics(rows,select['tau_BCTU'],indices)|{'evaluation_mode':modes[name]} for name,rows in methods.items()};reference=methods['adaptive_relative'];comparisons={name:paired(reference,rows,indices) for name,rows in methods.items() if name!='adaptive_relative' and not name.startswith('oracle')}
    family=[f'fixed_{r:.1f}' for r in RATIOS]+['best_global_fixed_dev','matched_mixture_seed17','length_heuristic','p1_difficulty_heuristic'];holm(comparisons,family);secondary=[k for k in comparisons if k not in family];holm(comparisons,secondary)
    for k in family:comparisons[k]['holm_family']='12_primary_references'
    for k in secondary:comparisons[k]['holm_family']='secondary_inference_ablations'
    rawtable={(x['id'],x['condition']):x for x in raw if x['experiment_split']=='screen'};mismatch=[q for q,j in zip(ids,primary_actions) if rawtable[q,'adaptive_relative']['token_ids']!=rawtable[q,f'fixed_{RATIOS[j]:.1f}']['token_ids']]
    pd=P[:,valid];yd=Y[:,valid];bins=[];ece=0
    for loedge in np.linspace(0,.9,10):
        upper=(pd<loedge+.1) if loedge<.89 else (pd<=1)
        mask=(pd>=loedge)&upper
        if not mask.any():continue
        pp=float(pd[mask].mean());yy=float(yd[mask].mean());count=int(mask.sum());ece+=count/pd.size*abs(pp-yy);bins.append({'lower':float(loedge),'n':count,'predicted':pp,'observed':yy})
    oracle_cost=TC[np.arange(n),oa];chosen_cost=np.array([x['output_tokens'] for x in reference]);chosen_y=np.array([x['correct'] for x in reference]);anysafe=((Y==1)&valid[None,:]).any(1)
    diag={'same_query_same_ratio_identity':{'n':256,'mismatches':mismatch,'passed':not mismatch},'nonmonotone_binary_correctness_fraction_primary_range':float((np.diff(Y[:,valid],axis=1)<0).any(1).mean()),'nonmonotone_binary_correctness_fraction_all_rates':float((np.diff(Y,axis=1)<0).any(1).mean()),'all_wrong_fraction_primary_range':float((~anysafe).mean()),'avoidable_error_fraction':float(((chosen_y==0)&anysafe).mean()),'safe_but_redundant_fraction':float(((chosen_y==1)&(chosen_cost>oracle_cost)).mean()),'mean_selected_token_gap_to_oracle':float((chosen_cost-oracle_cost).mean()),'calibration':{'Brier':float(np.mean((pd-yd)**2)),'ECE_equal_width10':ece,'AUROC':float(roc_auc_score(yd.ravel(),pd.ravel())),'AUPRC':float(average_precision_score(yd.ravel(),pd.ravel())),'bins':bins,'note':'Frozen Qwen3B predictions evaluated against Qwen7B generation; no target calibration fitted.'},'recipe_interpretation':'All checkpoints existed before this request. soft_cost removes boundary+policy block jointly; soft_mono_boundary adds soft monotonicity; asymmetric_soft_mono_boundary additionally changes negative BCE weight. No new training, and no K1-vs-K3 trained-model comparison.'}
    diag['correct_only_redundant_output_tokens']=float((chosen_cost-oracle_cost)[chosen_y==1].mean()) if (chosen_y==1).any() else None
    diag['oracle_gap_note']='The all-query selected-token gap can be negative on incorrect cheap answers; it is not a quality-matched saving.'
    seeds=[result[f'controller_seed{x}'] for x in SEEDS];seedsummary={k:{'mean':float(np.mean([x[k] for x in seeds])),'std_ddof1':float(np.std([x[k] for x in seeds],ddof=1))} for k in ['accuracy','mean_output_tokens','mean_paper_cot_tokens']}
    budgets={f'matched_mixture_seed{seed}':{'screen_token_gap_vs_adaptive':result[f'matched_mixture_seed{seed}']['mean_output_tokens']-result['adaptive_relative']['mean_output_tokens'],'relative_gap':result[f'matched_mixture_seed{seed}']['mean_output_tokens']/result['adaptive_relative']['mean_output_tokens']-1} for seed in SEEDS}
    writel(RUN/'policy_rows_screen.jsonl',[r for rows in methods.values() for r in rows]);dump(RUN/'METHOD_RESULTS.json',result);dump(RUN/'PAIRED_COMPARISONS.json',comparisons);dump(RUN/'DIAGNOSTICS.json',diag);dump(RUN/'SEED_STUDY.json',{'seeds':list(SEEDS),'new_training':False,'independent_previously_trained_controller_policies':True,'decoder_random_seeds_not_repeated_training':True,'statistics':seedsummary});dump(RUN/'BUDGET_GAPS.json',budgets)
    # Flat tables retain all methods, including unfavorable results and oracle flags.
    flat=[]
    for name,r in result.items():flat.append({'method':name,'evaluation_mode':r['evaluation_mode'],'n':r['n'],'correct':r['correct'],'accuracy_percent':100*r['accuracy'],'mean_output_tokens':r['mean_output_tokens'],'mean_paper_cot_tokens':r['mean_paper_cot_tokens'],'median_output_tokens':r['median_output_tokens'],'p90_output_tokens':r['p90_output_tokens'],'correct_per_1k_tokens':r['correct_per_1k_tokens'],'tokens_per_correct':r['tokens_per_correct'],'E3':r['E3'],'BCTU_beta1':r['BCTU']['1.0']})
    with (RUN/'method_results.csv').open('w',newline='') as file:w=csv.DictWriter(file,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    costledger={'offline_bank_generations':len(raw),'offline_total_input_tokens':sum(x['input_tokens'] for x in raw),'offline_total_output_tokens':sum(x['output_tokens'] for x in raw),'offline_generation_wall_seconds':sum(x['total_seconds'] for x in readl(RUN/'batches.jsonl')),'online_policy_generations_per_question':1,'candidate_bank_is_offline_experiment_not_deployment':True,'actual_adaptive_screen_output_tokens':result['adaptive_relative']['total_output_tokens'],'cached_policy_latency':'Not measured; report no invented per-policy latency. Separate B1 study required.','weight_training_this_stage':0}
    dump(RUN/'COST_LEDGER.json',costledger)
    done={'status':'analysis_complete','analysis_code_sha256':digest(__file__),'freeze_sha256':digest(FREEZE),'raw_sha256':digest(RUN/'raw.jsonl'),'scores_screen_sha256':digest(RUN/'scores_screen.jsonl'),'methods':len(result),'identity_passed':not mismatch,'primary_adaptive':result['adaptive_relative'],'primary_vs_r1':comparisons['fixed_1.0'],'primary_vs_matched_budget':comparisons['matched_mixture_seed17'],'main_policy_changed':False,'new_training':False,'remaining':'B1 overhead/latency, K3 inference-only diagnostic, final figures and package. Training ablations deliberately unavailable.'};dump(RUN/'ANALYSIS_COMPLETE.json',done);dump(RUN/'analysis_status.json',done);print(json.dumps(done),flush=True)
if __name__=='__main__':main()
