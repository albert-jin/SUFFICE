"""Bounded selection on existing dev data only; no calibration/reserve access."""
import argparse,itertools
from common import *
from dual_view_policy import choose_dual_view
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--out',required=True);ap.add_argument('--plan');a=ap.parse_args();data=Path(a.data_root);model=Path(a.model_dir);out=Path(a.out)
    inputs=readl(data/'data/development_pool.jsonl');dev=[x for x in inputs if x['split']=='dev'];assert len(dev)==900
    preds=[np.load(model/f'predictions_seed{s}.npz') for s in SEEDS];allids=preds[0]['ids'];assert all(np.array_equal(p['ids'],allids) for p in preds);lookup={i:k for k,i in enumerate(allids)};ix=[lookup[x['id']] for x in dev]
    sj=np.stack([p['joint'][ix] for p in preds]);sq=np.stack([p['recovery'][ix] for p in preds]);sc=np.stack([p['cost'][ix] for p in preds]);saved={x['id']:x for x in readl(model/'policy_predictions.jsonl')}
    Y=np.array([saved[x['id']]['candidate_correctness'] for x in dev]);C=np.array([saved[x['id']]['candidate_tokens'] for x in dev],float);domains=np.array([x['dataset'] for x in dev]);index=np.arange(len(dev));ref=Y[:,-1];cases=[]
    plan_path=Path(a.plan) if a.plan else Path(__file__).parent/'plan_revision_v4.json';plan=json.loads(plan_path.read_text());search=plan['search']
    for tau,eps,beta in itertools.product(search['risk_threshold'],search['recoverability_loss_limit'],search['shared_disagreement_multiplier']):
        config={'risk_threshold':tau,'recoverability_loss_limit':eps,'beta':beta,'min_ratio':.5}
        actions=choose_dual_view(sj,sq,sc,**config);y=Y[index,actions];cost=C[index,actions]
        delta=float((y-ref).mean());domain_delta={d:float((y-ref)[domains==d].mean()) for d in DATASETS};saving=float(1-cost.mean()/C[:,-1].mean())
        feasible=bool(delta>=0 and min(domain_delta.values())>=-.01 and saving>0)
        cases.append({'config':config,'accuracy':float(y.mean()),'mean_output_tokens':float(cost.mean()),'saving':saving,'delta_r1':delta,'domain_delta':domain_delta,'feasible':feasible,'compressed_fraction':float((actions!=7).mean())})
    feasible=[x for x in cases if x['feasible']]
    selected=min(feasible,key=lambda x:(x['mean_output_tokens'],-x['accuracy'],x['config']['beta'],x['config']['risk_threshold'],x['config']['recoverability_loss_limit'])) if feasible else None
    dump(out/'SEARCH.json',cases);report={'status':'selected_on_dev' if selected else 'no_feasible_policy','primary_checkpoint':model.name,'n':900,'reference_accuracy':float(ref.mean()),'reference_tokens':float(C[:,-1].mean()),'candidate_count':len(cases),'feasible_count':len(feasible),'selected':selected,'plan_sha256':digest(plan_path),'policy_code_sha256':digest(Path(__file__).parent/'dual_view_policy.py'),'calibration_labels_used':False,'reserve_labels_used':False,'model_files':{p.name:digest(p) for p in [model/'COMPLETE.json',model/'normalizer.npz',*[model/f'model_seed{s}.pt' for s in SEEDS]]}}
    dump(out/'SELECTION.json',report)
    if selected:
        actions=choose_dual_view(sj,sq,sc,**selected['config']);writel(out/'dev_predictions.jsonl',[{'id':x['id'],'dataset':x['dataset'],'selected_ratio':RATIOS[j],'selected_index':int(j),'correct':int(Y[i,j]),'output_tokens':float(C[i,j]),'reference_correct':int(ref[i])} for i,(x,j) in enumerate(zip(dev,actions))])
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
