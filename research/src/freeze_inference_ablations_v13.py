from inference_ablations_v13 import *
from sufficiency_model_v7 import SufficiencyRouter
from transformers import AutoTokenizer

def main():
    if FREEZE.exists():settings();return
    prior=ROOT/'manifests/QWEN7B_ADAPTIVE_V12_FREEZE.json';pf=json.loads(prior.read_text())
    for p,h in pf['source_files'].items():assert digest(ROOT/p)==h
    dev=[x for x in readl(ROOT/'data/dev.jsonl') if x['dataset']=='commonsenseqa'];screen=readl(ROOT/'data/llama_screen_v10s/commonsenseqa_inputs.jsonl')
    assert len(dev)==len(screen)==256 and not ({x['id'] for x in dev}&{x['id'] for x in screen})
    data=ROOT/'data/inference_ablations_v13';data.mkdir(exist_ok=True)
    writel(data/'dev_inputs.jsonl',[{k:v for k,v in x.items() if k!='answer'} for x in dev]);writel(data/'dev_targets.jsonl',[{'id':x['id'],'answer':x['answer']} for x in dev]);writel(data/'screen_inputs.jsonl',screen)
    # Already separated screening targets; copying preserves labels without using them.
    import shutil
    shutil.copyfile(ROOT/pf['screen_target_file'],data/'screen_targets.jsonl')
    arrays=np.load(ROOT/'data/shared_prefix_development_features.npz');lookup={str(i):j for j,i in enumerate(arrays['ids'])};dfeat=arrays['features'][[lookup[x['id']] for x in dev]].astype(np.float32)
    sid=[];sf=[]
    for p in sorted((ROOT/'runs/llama_transfer_v10s/routing/features').glob('commonsenseqa_*.npz')):
        meta=json.loads((p.parent.parent/'parts'/(p.stem+'.json')).read_text());assert digest(p)==meta['feature_sha256'];z=np.load(p);sid+=z['ids'].tolist();sf.extend(z['features'])
    assert sid==[x['id'] for x in screen];X=np.concatenate([dfeat,np.asarray(sf,np.float32)])
    ids=[x['id'] for x in dev]+sid;pred={'ids':np.array(ids),'split':np.array(['dev']*256+['screen']*256),'features':X};models={}
    for recipe in ['soft_boundary','soft_cost','soft_mono_boundary','asymmetric_soft_mono_boundary']:
        path=ROOT/'runs/v7_loss_search'/recipe;router=SufficiencyRouter(path,pf['policy_config']);xx=torch.tensor((X-router.mean)/router.std);P=[];C=[];Q=[]
        for seed,(net,w,smear) in zip(SEEDS,router.models):
            with torch.inference_mode():s,q,lc,_=net(xx)
            pp=(s+np.log(w)).sigmoid().numpy();cc=np.exp(lc.numpy().clip(0,9))*smear;P.append(pp);C.append(cc);Q.append(q.sigmoid().numpy())
            if recipe=='soft_boundary':pred[f'prob_seed{seed}']=pp;pred[f'cost_seed{seed}']=cc
            models[str((path/f'model_seed{seed}.pt').relative_to(ROOT))]=digest(path/f'model_seed{seed}.pt')
        pred['prob_'+recipe]=np.mean(P,0);pred['cost_'+recipe]=np.mean(C,0);pred['recover_'+recipe]=np.mean(Q,0)
        for name in ['normalizer.npz','COMPLETE.json']:models[str((path/name).relative_to(ROOT))]=digest(path/name)
    np.savez(data/'predictions.npz',**pred)
    acts=actions(pred['prob_soft_boundary'],pred['cost_soft_boundary']);tok=AutoTokenizer.from_pretrained(ADAPTER,local_files_only=True);widths={}
    for split,rows in [('dev',dev),('screen',screen)]:
        for i in range(0,256,8):widths[f'{split}_{i:04d}']=max(len(tok.encode(prompt(x['question'],r,tok),add_special_tokens=False)) for x in rows[i:i+8] for r in RATIOS)
    source={f'src/{n}':digest(ROOT/'src'/n) for n in ['inference_ablations_v13.py','freeze_inference_ablations_v13.py','generate_inference_ablations_v13.py','run_inference_ablations_v13.py','qwen7b_adaptive_v12.py','sufficiency_model_v7.py','common.py','score_bank.py']}
    source.update({str(p.relative_to(ROOT)):digest(p) for folder in ['eval','data_processing'] for p in (ROOT/'upstream'/folder).glob('*.py')})
    f={'version':'inference-only-ablations-v13','user_constraint':'No training of any model, adapter or controller; inference-only ablations.','base_model':pf['base_model'],'base_revision':pf['base_revision'],'adapter_repo':pf['adapter_repo'],'adapter_revision':pf['adapter_revision'],'parent_freeze_sha256':digest(prior),'source_files':source,'model_files':models,'assets_manifest_sha256':digest(ADAPTER/'ASSETS_MANIFEST.json'),'adapter_weights_sha256':digest(ADAPTER/'adapter_model.safetensors'),'data_files':{str(p.relative_to(ROOT)):digest(p) for p in data.glob('*') if p.is_file()},'question_counts':{'dev':256,'screen':256},'conditions':CONDITIONS,'generations_planned':512*len(CONDITIONS),'batch_size':8,'max_new_tokens':512,'physical_widths':widths,'primary_policy_config':pf['policy_config'],'primary_min_ratio':.5,'fixed_0p3_0p4_scope':'Extra diagnostic fixed baselines outside primary policy range; not allowed primary actions.','numerics':{'dtype':'BF16','attention':'SDPA','tf32':False,'allow_bf16_reduced_precision_reduction':False,'deterministic_algorithms':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8','padding':'per-question-batch max token width across every candidate ratio, reused for all methods','reason':'V12 r1 subset exhibited numerical differences when other rows changed padding width. This is a separate corrected-runtime experiment; keep V11/V12 records.'},'analysis_plan':{'fixed_baselines':list(RATIOS),'best_global_fixed':'Choose on dev: among fixed r with dev accuracy >= fixed r1, minimize actual mean output tokens; otherwise r1.','matched_budget':'Freeze two-adjacent-r mixture and probability from DEV actual costs to match primary DEV mean output tokens; seed17 primary, seeds43/79 sensitivity. Report realized screen budget gaps; never select on screen labels.','heuristics':['length quantiles with primary dev action histogram','p1 difficulty quantiles with primary dev action histogram'],'strategy_ablations':['absolute_tau0.97','without_correctness_gate','without_learned_cost','shuffled_primary_actions_seeds17_43_79'],'existing_checkpoints':['single_seed17','single_seed43','single_seed79','soft_cost (boundary+policy block removed)','soft_mono_boundary (existing monotonic-loss checkpoint)','asymmetric_soft_mono_boundary (existing asymmetric-BCE checkpoint)'],'unavailable_training_ablations':'No K1/vote/raw-soft/Jeffreys retraining, no new loss training. These cannot be claimed completed.','statistics':'paired question bootstrap10000, exact McNemar, Holm correction across eight fixed baselines plus dev-selected global, primary matched mixture and two heuristics; auxiliary ablations separate family.','diagnostics':['real accuracy nonmonotonicity','min-actual-token-correct oracle (not deployable)','under/over-allocation','ECE/Brier/AUROC','same-r bank versus actual adaptive token identity','median/P90 output','all offline candidate generation cost separate from one-generation policy cost']},'future_inference_only_supplements':['B1 timing on64 frozen dev questions with encoder overhead, no training','K3 recoverability diagnostic on64 frozen dev questions, question-only/true-CoT/mismatched-CoT controls; same CoT three sampled answers, not deployment-time search'],'benchmark_scope':'Prior CSQA dev and reused screen only; old results observed, exploratory evidence not fresh final testing. Original-paper adapter differs. No other domain reruns.','training_performed':False,'main_policy_tuned':False,'published_baselines_file':pf['published_baselines_file'],'published_baselines_sha256':pf['published_baselines_sha256']}
    dump(FREEZE,f);print(json.dumps({'status':'frozen','generations':f['generations_planned'],'freeze_sha256':digest(FREEZE)}),flush=True)
if __name__=='__main__':main()
