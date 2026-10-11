import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import fcntl,subprocess,traceback
from inference_ablations_v13 import *
from shared_decode import SharedDecoder
from sufficiency_model_v7 import SufficiencyRouter

def main():
    run=RUN/'latency_b1';run.mkdir(exist_ok=True);lock=(run/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'COMPLETE.json').exists():return
    f=settings();assert (RUN/'ANALYSIS_COMPLETE.json').exists();sel=json.loads((RUN/'DEV_BASELINE_SELECTION.json').read_text());mix=sel['matched_mixture']
    inputs=readl(ROOT/'data/inference_ablations_v13/dev_inputs.jsonl');items=sorted(inputs,key=lambda x:hashlib.sha256(('latency-v13|'+x['id']).encode()).hexdigest())[:64]
    frozen={'n':64,'ids':[x['id'] for x in items],'source':'pre-existing CSQA dev only','methods':['fixed_r1','adaptive_relative','matched_mixture_seed17'],'decode':'B1 greedy BF16, same per-question width across all rates, cap512','timing':'Interleaved method order per question; total includes width planning, encoder+router only for adaptive, tokenization and one generator call. Both models resident; loading/warmup recorded separately. No fit.','main_freeze_sha256':digest(FREEZE),'selection_sha256':digest(RUN/'DEV_BASELINE_SELECTION.json'),'code_sha256':digest(__file__),'new_training':False};dump(run/'FREEZE.json',frozen)
    jobs=[]
    for i,x in enumerate(items):
        order=frozen['methods'][i%3:]+frozen['methods'][:i%3]
        for m in order:jobs.append((f'{i:03d}_{m}',x,m))
    missing=[j for j in jobs if not (run/'parts'/(j[0]+'.json')).exists()];completed=len(jobs)-len(missing)
    try:
        if missing:
            assert float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0])>22000
            torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
            t=time.perf_counter();engine=MatchedWidthGenerator();encoder=SharedDecoder();encoder.model.requires_grad_(False);router=SufficiencyRouter(ROOT/f['primary_controller_path'] if 'primary_controller_path' in f else ROOT/'runs/v7_loss_search/soft_boundary',f['primary_policy_config']);torch.cuda.synchronize();load=time.perf_counter()-t
            warm=readl(ROOT/'data/train.jsonl')[0]['question'];b=encoder.prefill([warm]);router.route(b['features']);del b
            width=max(len(engine.tokenizer.encode(prompt(warm,r,engine.tokenizer),add_special_tokens=False)) for r in RATIOS);wo,wt=engine.generate_width([warm],[1.],width,16)
            dump(run/'attempts'/f'{int(time.time())}.json',{'dual_model_load_seconds':load,'warmup_output':wo,'warmup_timing':wt,'warmup_not_evaluation':True,'loaded_gpu_allocated_bytes':torch.cuda.memory_allocated()})
        for key,x,method in missing:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();plan_start=start
            width=max(len(engine.tokenizer.encode(prompt(x['question'],r,engine.tokenizer),add_special_tokens=False)) for r in RATIOS);planning=time.perf_counter()-plan_start;encsecs=0.;routesecs=0.;enc_tokens=0;policy=None
            if method=='adaptive_relative':
                batch=encoder.prefill([x['question']]);encsecs=batch['prefill_seconds']+batch['plan_seconds'];enc_tokens=batch['prefix_tokens'][0];rt=time.perf_counter();action=router.route(batch['features'])[0];routesecs=time.perf_counter()-rt;ratio=float(RATIOS[action]);policy=router.last[0];del batch
            elif method=='fixed_r1':ratio=1.
            else:
                u=int.from_bytes(hashlib.sha256(f"17|{x['id']}".encode()).digest()[:8],'big')/2**64;ratio=mix['high_ratio'] if u<mix['probability_high'] else mix['low_ratio']
            torch.manual_seed(20261010);torch.cuda.manual_seed_all(20261010);rows,times=engine.generate_width([x['question']],[ratio],width,512);torch.cuda.synchronize();total=time.perf_counter()-start
            row={'id':x['id'],'method':method,**rows[0],'total_request_seconds':total,'width_planning_seconds':planning,'encoder_seconds':encsecs,'router_seconds':routesecs,'generator_seconds':times['total_seconds'],'encoder_input_tokens':enc_tokens,'policy':policy,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'physical_width':width}
            dump(run/'parts'/(key+'.json'),row);completed+=1;status={'status':'measuring','completed_requests':completed,'total_requests':192,'new_training':False};dump(run/'status.json',status);print(json.dumps(status),flush=True)
        raw=[json.loads((run/'parts'/(key+'.json')).read_text()) for key,_,_ in jobs];writel(run/'raw.jsonl',raw)
        summary={}
        for m in frozen['methods']:
            rr=[r for r in raw if r['method']==m];v=np.array([r['total_request_seconds'] for r in rr]);summary[m]={'n':len(rr),'mean_seconds':float(v.mean()),'median_seconds':float(np.median(v)),'p95_seconds':float(np.quantile(v,.95)),'mean_output_tokens':float(np.mean([r['output_tokens'] for r in rr])),'mean_encoder_seconds':float(np.mean([r['encoder_seconds'] for r in rr])),'mean_router_seconds':float(np.mean([r['router_seconds'] for r in rr])),'mean_generator_seconds':float(np.mean([r['generator_seconds'] for r in rr])),'max_allocated_bytes':max(r['peak_allocated_bytes'] for r in rr)}
        lookup={(r['id'],r['method']):r for r in raw};a=np.array([lookup[x['id'],'adaptive_relative']['total_request_seconds'] for x in items]);rng=np.random.default_rng(20261010);ix=rng.integers(64,size=(10000,64));paired={}
        for m in ['fixed_r1','matched_mixture_seed17']:
            b=np.array([lookup[x['id'],m]['total_request_seconds'] for x in items]);paired[m]={'mean_latency_delta_seconds':float((a-b).mean()),'paired_ci95':np.quantile((a-b)[ix].mean(1),[.025,.975]).tolist(),'ratio_of_mean_latencies':float(a.mean()/b.mean())}
        done={'status':'latency_complete','summary':summary,'paired':paired,'raw_sha256':digest(run/'raw.jsonl'),'n_queries':64,'actual_requests':192,'model_load_excluded_from_steady_state':True,'warmup_and_load_preserved':True,'note':'DEV subset, B1, one measured trial per query/method with rotated order. Primary adaptive computes new B1 query features; numerical choices may differ from stored B16 features. Not a full-population or repeated-run speed claim.','new_training':False};dump(run/'COMPLETE.json',done);dump(run/'status.json',done);print(json.dumps(done),flush=True)
    except BaseException:dump(run/'status.json',{'status':'failed','completed_requests':completed,'error':traceback.format_exc()});raise
if __name__=='__main__':main()
