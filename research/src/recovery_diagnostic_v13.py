import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import signal,fcntl,subprocess,traceback
from inference_ablations_v13 import *
from score_bank import grade,timeout_handler

from transformers import TemperatureLogitsWarper, TopPLogitsWarper

class CpuTopPLogitsWarper:
    """Use official HF nucleus filter on CPU; CUDA cumsum is unsupported in strict deterministic mode."""
    def __init__(self, top_p): self.filter = TopPLogitsWarper(top_p)
    def __call__(self, input_ids, scores):
        return self.filter(input_ids.cpu(), scores.cpu()).to(scores.device)

def main():
    run=RUN/'recovery_k3';run.mkdir(exist_ok=True);lock=(run/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'COMPLETE.json').exists():return
    f=settings();assert (RUN/'ANALYSIS_COMPLETE.json').exists();inputs=readl(ROOT/'data/inference_ablations_v13/dev_inputs.jsonl');items=sorted(inputs,key=lambda x:hashlib.sha256(('recovery-v13|'+x['id']).encode()).hexdigest())[:64]
    raw={(r['id'],r['condition']):r for r in readl(RUN/'raw.jsonl') if r['experiment_split']=='dev'};rates=[.5,.7,1.];prompts=[]
    for i,x in enumerate(items):
        prompts.append({'id':x['id'],'condition':'question_only','ratio':None,'cot_source_id':None,'cot':'','prompt':recovery_prompt(x['question'],'')})
        for r in rates:
            for cond,source in [('true_cot',x['id']),('mismatched_cot',items[(i+17)%64]['id'])]:
                cot,cut=strip_final(raw[source,f'fixed_{r:.1f}']['text']);prompts.append({'id':x['id'],'condition':cond,'ratio':r,'cot_source_id':source,'cot':cot,'explicit_answer_cut':cut,'prompt':recovery_prompt(x['question'],cot)})
    plan={'n_questions':64,'question_ids':[x['id'] for x in items],'rates':rates,'primary_K':3,'primary_seeds':[17,43,79],'optional_true_cot_K5_seeds':[17,43,79,97,131],'temperature':.7,'top_p':.95,'nucleus_runtime':'HF TopPLogitsWarper on CPU; strict deterministic mode; explicit temperature processor before nucleus; built-in sampling temperature=1/top_p=1 disables duplicate processing','top_k':0,'max_new_tokens':32,'model':'unadapted Qwen2.5-7B-Instruct fixed base','CoT_source':'same fixed-r generated CoT from V13 dev, never regenerate it across recovery samples','mismatch':'fixed cyclic offset17 across the64 predetermined question IDs, within each retention','roles':'annotation and faithfulness diagnostics only, not deployment inference and not model training','main_freeze_sha256':digest(FREEZE),'code_sha256':digest(__file__)};dump(run/'FREEZE.json',plan);writel(run/'prompts.jsonl',[{**p,'prompt_sha256':hashlib.sha256(p['prompt'].encode()).hexdigest(),'cot_sha256':hashlib.sha256(p['cot'].encode()).hexdigest()} for p in prompts])
    jobs=[]
    for cond in ['question_only','true_cot','mismatched_cot']:
        for r in ([None] if cond=='question_only' else rates):
            pool=[p for p in prompts if p['condition']==cond and p['ratio']==r];seeds=plan['optional_true_cot_K5_seeds'] if cond=='true_cot' else plan['primary_seeds']
            for start in range(0,64,16):
                for seed in seeds:jobs.append((f'{cond}_{r}_{start:03d}_{seed}',cond,r,seed,pool[start:start+16]))
    missing=[j for j in jobs if not (run/'parts'/(j[0]+'.json')).exists()];completed=sum(len(p) for k,c,r,s,p in jobs if (run/'parts'/(k+'.json')).exists())
    try:
        if missing:
            assert float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])>22000
            from transformers import AutoModelForCausalLM,AutoTokenizer
            torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
            tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);tok.padding_side='left';start=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).cuda().eval().requires_grad_(False);torch.cuda.synchronize();dump(run/'attempts'/f'{int(time.time())}.json',{'load_seconds':time.perf_counter()-start,'adapter_applied':False,'training':False})
        for key,cond,r,seed,rows in missing:
            effective=int.from_bytes(hashlib.sha256(f'recovery-v13|{key}'.encode()).digest()[:4],'big');torch.manual_seed(effective);torch.cuda.manual_seed_all(effective);start=time.perf_counter();enc=tok([p['prompt'] for p in rows],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda');assert enc.input_ids.shape[1]+32<=8192
            torch.cuda.synchronize()
            with torch.inference_mode():out=model.generate(**enc,max_new_tokens=32,do_sample=True,temperature=1.,top_p=1.,top_k=0,logits_processor=[TemperatureLogitsWarper(.7),CpuTopPLogitsWarper(.95)],eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start;records=[]
            for p,seq,ninput in zip(rows,out[:,enc.input_ids.shape[1]:].tolist(),enc.attention_mask.sum(1).tolist()):
                stop=tok.eos_token_id in seq
                if stop:seq=seq[:seq.index(tok.eos_token_id)+1]
                records.append({'id':p['id'],'condition':cond,'ratio':r,'seed':seed,'effective_batch_seed':effective,'cot_source_id':p['cot_source_id'],'cot_sha256':hashlib.sha256(p['cot'].encode()).hexdigest(),'prompt_sha256':hashlib.sha256(p['prompt'].encode()).hexdigest(),'text':tok.decode(seq,skip_special_tokens=True),'token_ids':seq,'input_tokens':ninput,'output_tokens':len(seq),'truncated':not stop})
            dump(run/'parts'/(key+'.json'),{'records':records,'wall_seconds':elapsed,'base_seed':seed,'effective_batch_seed':effective});completed+=len(records);status={'status':'sampling_recovery','completed_responses':completed,'total_responses':1728,'new_training':False};dump(run/'status.json',status);print(json.dumps(status),flush=True)
        records=[];wall=0
        for key,*_ in jobs:
            d=json.loads((run/'parts'/(key+'.json')).read_text());records+=d['records'];wall+=d['wall_seconds']
        assert len(records)==1728;writel(run/'raw.jsonl',records)
        # Labels are opened only after sampling is complete; no learning or selection follows.
        gold={r['id']:r['answer'] for r in readl(ROOT/'data/inference_ablations_v13/dev_targets.jsonl')};lookup={x['id']:{**x,'answer':gold[x['id']]} for x in items};signal.signal(signal.SIGALRM,timeout_handler);scores=[]
        for rec in records:
            answer,y,error=grade(lookup[rec['id']],rec['text']);assert error is None,(rec['id'],error);scores.append({k:rec[k] for k in ['id','condition','ratio','seed','output_tokens','input_tokens','truncated','cot_sha256','prompt_sha256']}|{'correct':y,'prediction':answer})
        writel(run/'scores.jsonl',scores);table={(s['id'],s['condition'],s['ratio'],s['seed']):s for s in scores};labels=[]
        for x in items:
            for r in rates:
                rr=[table[x['id'],'true_cot',r,s] for s in [17,43,79,97,131]];assert len({p['cot_sha256'] for p in rr})==len({p['prompt_sha256'] for p in rr})==1;c3=sum(z['correct'] for z in rr[:3]);c5=sum(z['correct'] for z in rr)
                labels.append({'id':x['id'],'ratio':r,'correct3':c3,'raw_soft3':c3/3,'jeffreys_soft3':(c3+.5)/4,'majority3':int(c3>=2),'correct5':c5,'raw_soft5':c5/5,'jeffreys_soft5':(c5+.5)/6,'single_labels_by_choice_seed':{str(seed):rr[int.from_bytes(hashlib.sha256(f'{seed}|{x["id"]}|{r}'.encode()).digest()[:4],'big')%3]['correct'] for seed in [17,43,79]}})
        writel(run/'soft_labels.jsonl',labels);rng=np.random.default_rng(20261010);ix=rng.integers(64,size=(10000,64));summary={};conditions=['question_only','true_cot','mismatched_cot']
        for r in rates:
            vectors={c:np.array([np.mean([table[x['id'],c,None if c=='question_only' else r,s]['correct'] for s in [17,43,79]]) for x in items]) for c in conditions}
            summary[str(r)]={'mean_recovery_accuracy':{c:float(v.mean()) for c,v in vectors.items()},'true_minus_question_only':{'delta':float((vectors['true_cot']-vectors['question_only']).mean()),'paired_question_ci95':np.quantile((vectors['true_cot']-vectors['question_only'])[ix].mean(1),[.025,.975]).tolist()},'true_minus_mismatched':{'delta':float((vectors['true_cot']-vectors['mismatched_cot']).mean()),'paired_question_ci95':np.quantile((vectors['true_cot']-vectors['mismatched_cot'])[ix].mean(1),[.025,.975]).tolist()}}
        done={'status':'recovery_diagnostic_complete','n_questions':64,'primary_same_CoT_samples':3,'true_CoT_extra_K5_sensitivity':True,'total_responses':1728,'total_input_tokens':sum(x['input_tokens'] for x in records),'total_output_tokens':sum(x['output_tokens'] for x in records),'offline_wall_seconds':wall,'truncated_responses':sum(x['truncated'] for x in records),'fraction_CoTs_with_nonunanimous_three_answers':float(np.mean([0<x['correct3']<3 for x in labels])),'per_ratio_controls':summary,'raw_sha256':digest(run/'raw.jsonl'),'soft_labels_sha256':digest(run/'soft_labels.jsonl'),'model_training':False,'interpretation':'Annotation stability and CoT-use diagnostic only; does not test whether training with K3 beats K1. All extra responses are offline cost, not part of deployed policy.'};dump(run/'COMPLETE.json',done);dump(run/'status.json',done);print(json.dumps(done),flush=True)
    except BaseException:dump(run/'status.json',{'status':'failed','completed_responses':completed,'error':traceback.format_exc()});raise
if __name__=='__main__':main()
