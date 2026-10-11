"""Resumable offline retention/recovery banks. Never an online all-ratio policy."""
import os,time,argparse,json,sys,signal,subprocess
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['TOKENIZERS_PARALLELISM']='false'
from common import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--run',required=True);ap.add_argument('--mode',choices=['retention','recovery','controls'],required=True);ap.add_argument('--source');ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    root=ROOT;run=root/'runs'/args.run;run.mkdir(parents=True,exist_ok=True)
    rows=readl(args.input)
    if args.limit:rows=rows[:args.limit]
    if 'sealed_test' in str(args.input) and not (root/'manifests/FINAL_FREEZE.json').exists():raise RuntimeError('Final test requires a frozen configuration')
    cfg={'input_sha256':digest(args.input),'source_sha256':digest(args.source) if args.source else None,'mode':args.mode,'ratios':list(RATIOS),'answer_seeds':list(SEEDS),'answer_temperature':.7,'answer_top_p':.95,'answer_max_tokens':128,'retention_temperature':0,'retention_seed':42,'cap_rule':'GSM8K/MCQ 512; MATH 1024 for every ratio; paper ratio-adjusted caps are a separate later diagnostic','generator':'same frozen merged Qwen2.5-3B TokenSkip','recoverer':'same frozen Qwen2.5-3B base, without adapter','n':len(rows),'max_num_seqs':64,'offline_only':True}
    if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==cfg,'Cannot change a resumed run'
    else:dump(run/'config.json',cfg)
    outpath=run/'bank.jsonl';completed={x['key'] for x in readl(outpath)} if outpath.exists() else set()
    if (run/'COMPLETE.json').exists():print('ALREADY_COMPLETE',args.run);return
    requests=[];byid={x['id']:x for x in rows}
    if args.mode=='retention':
        for x in rows:
            for ratio in RATIOS:
                key=f"{x['id']}|r{ratio:.1f}"
                requests.append({'key':key,'id':x['id'],'dataset':x['dataset'],'ratio':ratio,'seed':42,'prompt':make_prompt(x['question'],ratio),'max_tokens':1024 if x['dataset']=='math500' else 512})
    else:
        source=[z for z in readl(args.source) if z['id'] in byid]
        source.sort(key=lambda z:(z['dataset'],z['id'],z['ratio']))
        pools={d:[z for z in source if z['dataset']==d] for d in DATASETS}
        for z in source:
            x=byid[z['id']];cot,removed=strip_final(z['text'])
            if args.mode=='recovery':conditions=[('true',cot,None)]
            else:
                # Controls use fixed 0.5/1.0 ratios and first 32 dev queries/domain.
                eligible=[a['id'] for a in rows if a['dataset']==x['dataset'] and a.get('split')=='dev'][:32]
                if x['id'] not in eligible or z['ratio'] not in (.5,1.):continue
                pool=pools[x['dataset']];j=next(i for i,a in enumerate(pool) if a['key']==z['key'])
                mismatch=next(pool[(j+k)%len(pool)] for k in range(1,len(pool)) if pool[(j+k)%len(pool)]['id']!=z['id'] and pool[(j+k)%len(pool)]['ratio']==z['ratio'])
                other,_=strip_final(mismatch['text']);conditions=[('question_only','',None),('mismatch',other,mismatch['id'])]
            for condition,context,otherid in conditions:
                prompt=recovery_prompt(x['question'],context)
                for seed in SEEDS:
                    key=f"{z['key']}|{condition}|s{seed}"
                    requests.append({'key':key,'id':x['id'],'dataset':x['dataset'],'ratio':z['ratio'],'seed':seed,'condition':condition,'cot_sha256':hashlib.sha256(context.encode()).hexdigest(),'source_key':z['key'],'mismatch_source_id':otherid,'explicit_answer_removed':removed,'prompt':prompt,'max_tokens':128})
    missing=[q for q in requests if q['key'] not in completed]
    if not missing:
        dump(run/'COMPLETE.json',{'expected':len(requests),'n':len(completed),'bank_sha256':digest(outpath)});return
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()
    if float(gpu[0])<21000:raise RuntimeError('GPU occupied; do not interfere with other projects')
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    import torch,transformers,vllm
    model=root/'models'/('merged' if args.mode=='retention' else 'base')
    tok=AutoTokenizer.from_pretrained(str(root/'models/base'),local_files_only=True)
    start=time.time();llm=LLM(model=str(model),tokenizer=str(root/'models/base'),dtype='bfloat16',gpu_memory_utilization=.82,max_model_len=4096,max_num_seqs=64,max_num_batched_tokens=4096,enforce_eager=True,enable_prefix_caching=True,seed=42,disable_log_stats=True)
    init_seconds=time.time()-start
    dump(run/'runtime.json',{'torch':torch.__version__,'transformers':transformers.__version__,'vllm':vllm.__version__,'init_seconds':init_seconds,'code_sha256':digest(__file__)})
    batches=run/'batches.jsonl';new_in=new_out=0
    with outpath.open('a',encoding='utf-8',buffering=1) as f,batches.open('a',encoding='utf-8',buffering=1) as bf:
        for offset in range(0,len(missing),64):
            batch=missing[offset:offset+64];params=[SamplingParams(temperature=0 if args.mode=='retention' else .7,top_p=1 if args.mode=='retention' else .95,max_tokens=q['max_tokens'],seed=q['seed'],stop=[tok.eos_token]) for q in batch]
            t=time.perf_counter();outs=llm.generate([q['prompt'] for q in batch],params,use_tqdm=False);wall=time.perf_counter()-t
            for q,o in zip(batch,outs):
                v=o.outputs[0];rec={**q,'text':v.text,'token_ids':v.token_ids,'input_tokens':len(o.prompt_token_ids),'output_tokens':len(v.token_ids),'finish_reason':v.finish_reason,'truncated':v.finish_reason=='length','batch_index':offset//64,'batch_wall_seconds':wall,'batch_size':len(batch)}
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');new_in+=rec['input_tokens'];new_out+=rec['output_tokens'];completed.add(q['key'])
            bf.write(json.dumps({'start_offset':offset,'n':len(batch),'wall_seconds':wall})+'\n')
            stat={'stage':args.mode,'run':args.run,'completed':len(completed),'expected':len(requests),'wall_seconds_this_invocation':time.time()-start,'new_input_tokens':new_in,'new_output_tokens':new_out}
            dump(run/'status.json',stat);print(json.dumps(stat),flush=True)
    assert len(completed)==len(requests)
    dump(run/'COMPLETE.json',{**stat,'bank_sha256':digest(outpath),'init_seconds':init_seconds})
if __name__=='__main__':main()

