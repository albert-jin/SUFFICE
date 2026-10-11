import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import fcntl,traceback,subprocess
from inference_ablations_v13 import *

def main():
    f=settings();fh=digest(FREEZE);lock=(RUN/'generation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (RUN/'BANK_COMPLETE.json').exists():return
    for p,h in f['data_files'].items():assert digest(ROOT/p)==h
    pred=np.load(ROOT/'data/inference_ablations_v13/predictions.npz');act=actions(pred['prob_soft_boundary'],pred['cost_soft_boundary']);amap={str(i):float(RATIOS[j]) for i,j in zip(pred['ids'],act)}
    jobs=[]
    for split in ['dev','screen']:
        rows=readl(ROOT/f'data/inference_ablations_v13/{split}_inputs.jsonl');assert len(rows)==256 and all('answer' not in x for x in rows)
        for i in range(0,256,8):
            # Rotate order to avoid assigning all cold/late timing to one method.
            order=CONDITIONS[i//8%len(CONDITIONS):]+CONDITIONS[:i//8%len(CONDITIONS)]
            for cond in order:jobs.append((f'{split}_{i:04d}_{cond}',split,i,cond,rows[i:i+8]))
    missing=[j for j in jobs if not (RUN/'parts'/(j[0]+'.json')).exists()];completed=sum(len(x) for k,s,i,c,x in jobs if (RUN/'parts'/(k+'.json')).exists());attempt=str(int(time.time()))
    try:
        if missing:
            assert digest(ADAPTER/'adapter_model.safetensors')==f['adapter_weights_sha256']
            free=float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits']).decode().splitlines()[0]);assert free>22000,'Do not overlap GPU work'
            torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
            dump(RUN/'status.json',{'status':'loading','completed_generations':completed,'total_generations':len(jobs)*8})
            engine=MatchedWidthGenerator();assert not engine.model.training and not any(p.requires_grad for p in engine.model.parameters())
            dump(RUN/'attempts'/f'{attempt}.json',{'load_merge_seconds':engine.load_seconds,'training':False,'torch':torch.__version__,'cuda':torch.version.cuda,'deterministic':torch.are_deterministic_algorithms_enabled()})
        for key,split,i,cond,items in missing:
            ratios=[amap[x['id']] for x in items] if cond=='adaptive_relative' else [float(cond.removeprefix('fixed_'))]*len(items)
            torch.manual_seed(20261010);torch.cuda.manual_seed_all(20261010)
            out,timing=engine.generate_width([x['question'] for x in items],ratios,f['physical_widths'][f'{split}_{i:04d}'],512)
            records=[{'id':x['id'],'dataset':'commonsenseqa','experiment_split':split,'condition':cond,'batch_key':key,**r} for x,r in zip(items,out)]
            dump(RUN/'parts'/(key+'.json'),{'key':key,'split':split,'condition':cond,'freeze_sha256':fh,'records':records,'timing':timing,'attempt':attempt})
            completed+=len(records);s={'status':'generating_bank','experiment_split':split,'condition':cond,'completed_generations':completed,'total_generations':f['generations_planned'],'offline_bank_not_online_search':True};dump(RUN/'status.json',s);print(json.dumps(s),flush=True)
        raw=[];batches=[]
        for key,split,i,cond,items in jobs:
            p=json.loads((RUN/'parts'/(key+'.json')).read_text());assert p['freeze_sha256']==fh and [x['id'] for x in p['records']]==[x['id'] for x in items];raw+=p['records'];batches.append({'key':key,'split':split,'condition':cond,**p['timing']})
        assert len(raw)==f['generations_planned'];writel(RUN/'raw.jsonl',raw);writel(RUN/'batches.jsonl',batches)
        done={'status':'bank_complete','records':len(raw),'raw_sha256':digest(RUN/'raw.jsonl'),'batches_sha256':digest(RUN/'batches.jsonl'),'freeze_sha256':fh,'target_labels_loaded':False,'offline_total_output_tokens':sum(x['output_tokens'] for x in raw)};dump(RUN/'BANK_COMPLETE.json',done);dump(RUN/'status.json',done)
    except BaseException:
        dump(RUN/'status.json',{'status':'failed','completed_generations':completed,'total_generations':f['generations_planned'],'error':traceback.format_exc(),'inflight_unsaved_tokens':'unknown'});raise
if __name__=='__main__':main()
