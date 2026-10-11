"""Select loss configuration and reliability threshold on GSM development only."""
import itertools
from common import *
from sufficiency_model_v7 import choose_sufficient
from train_sufficiency_v7 import CONFIGS

def main():
    root=ROOT/'runs/v7_loss_search';assert (root/'TRAINING_COMPLETE.json').exists()
    rows=readl(ROOT/'runs/dev_v5_shared_joint_smooth/policy_predictions.jsonl');dev=np.array([x['split']=='dev' for x in rows]);ds=np.array([x['dataset'] for x in rows])[dev]
    assert dev.sum()==900 and all(x['split'] in ['train','dev'] for x in rows)
    Y=np.array([x['candidate_correctness'] for x in rows])[dev];C=np.array([x['candidate_tokens'] for x in rows])[dev];ii=np.arange(len(C));gsm=ds=='gsm8k';allcases=[]
    def result(action):
        y=Y[ii,action];c=C[ii,action]
        return {d:{'n':int((mask:=ds==d).sum()),'accuracy':float(y[mask].mean()),'mean_output_tokens':float(c[mask].mean()),'accuracy_delta_r1':float((y-Y[:,-1])[mask].mean()),'saving_vs_r1':float(1-c[mask].mean()/C[mask,-1].mean()),'ratio_counts':{str(r):int((action[mask]==j).sum()) for j,r in enumerate(RATIOS)}} for d in DATASETS}
    for cfg in CONFIGS:
        path=root/cfg['name'];pred=[np.load(path/f'predictions_seed{s}.npz') for s in SEEDS]
        assert all(list(x['ids'])==[r['id'] for r in rows] for x in pred)
        probs=np.stack([x['prob'][dev] for x in pred]);p=probs.mean(0);c=np.stack([x['cost'][dev] for x in pred]).mean(0);u=probs.std(0)
        for tau,slack in itertools.product([.55,.65,.75,.85],[.03,.08,.15,.3]):
            policy={'tau':tau,'reference_slack':slack,'beta':0.,'min_ratio':.5};actions=choose_sufficient(p,c,policy,u);scores=result(actions);g=scores['gsm8k']
            case={'model':str(path.relative_to(ROOT)),'loss_config':cfg,'policy_config':policy,'dev_by_dataset':scores,
                'qualified_first_screen':g['accuracy_delta_r1']>=-.02 and g['saving_vs_r1']>=.15,
                'probability_brier_dev':float(((p-Y)**2).mean()),'probability_brier_gsm_dev':float(((p[gsm]-Y[gsm])**2).mean())}
            allcases.append(case)
    qualified=[x for x in allcases if x['qualified_first_screen']]
    if qualified:
        best=min(qualified,key=lambda x:(x['dev_by_dataset']['gsm8k']['mean_output_tokens'],-x['dev_by_dataset']['gsm8k']['accuracy']));status='selected_for_gsm_screen'
    else:
        candidates=[x for x in allcases if x['dev_by_dataset']['gsm8k']['saving_vs_r1']>=.1]
        best=max(candidates or allcases,key=lambda x:(x['dev_by_dataset']['gsm8k']['accuracy'],x['dev_by_dataset']['gsm8k']['saving_vs_r1']));status='exploratory_no_qualified_dev_candidate'
    model=ROOT/best['model'];files=[model/'COMPLETE.json',model/'normalizer.npz',*[model/f'model_seed{s}.pt' for s in SEEDS]]
    selected={'version':'sufficiency_v7','status':status,'selected':best,'model_files':{p.name:digest(p) for p in files},'candidates':len(allcases),'qualified_candidates':len(qualified),
        'selection_scope':'6143 training rows; loss/model/threshold selection on GSM256 dev. Other dev domain metrics reported, not optimized. No official test labels.',
        'limits':'Development replay and threshold/model search, not independent evidence. Three heads are an ensemble, not three independent online evaluations.',
        'changes':'Nonasymmetric/asymmetric correctness BCE, K3 Jeffreys soft auxiliary, soft monotonicity, actual-cost boundary ranking, paired relative-cost supervision and differentiable cost/error policy loss.'}
    dump(root/'SEARCH.json',allcases);dump(root/'SELECTION.json',selected);print(json.dumps(selected),flush=True)
if __name__=='__main__':main()
