"""Train a small, preregistered loss grid on existing train/dev banks only."""
import copy,time
import torch
from torch import nn
from common import *
from sufficiency_model_v7 import SufficiencyNet

CONFIGS=[
    {'name':'soft_cost','negative_weight':1.,'soft_weight':.25,'mono_weight':0.,'boundary_weight':0.,'policy_weight':0.},
    {'name':'soft_boundary','negative_weight':1.,'soft_weight':.25,'mono_weight':0.,'boundary_weight':.5,'policy_weight':.1},
    {'name':'soft_mono_boundary','negative_weight':1.,'soft_weight':.25,'mono_weight':.1,'boundary_weight':.5,'policy_weight':.1},
    {'name':'asymmetric_soft_mono_boundary','negative_weight':2.,'soft_weight':.25,'mono_weight':.1,'boundary_weight':.5,'policy_weight':.1},
]

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'runs/v7_loss_search';out.mkdir(exist_ok=True)
    rows=readl(ROOT/'runs/dev_v5_shared_joint_smooth/policy_predictions.jsonl')
    assert len(rows)==7043 and all(x['split'] in ['train','dev'] for x in rows)
    train=np.array([x['split']=='train' for x in rows]);dev=~train;assert train.sum()==6143 and dev.sum()==900
    feat=np.load(ROOT/'data/shared_prefix_development_features.npz');assert list(feat['ids'])==[x['id'] for x in rows]
    X=feat['features'].astype(np.float32);mean=X[train].mean(0);std=X[train].std(0).clip(.05);X=(X-mean)/std
    Y=np.array([x['candidate_correctness'] for x in rows],np.float32);Q=np.array([x['actual_recoverability'] for x in rows],np.float32)
    C=np.array([x['candidate_tokens'] for x in rows],np.float32);assert np.isin(Q,[.125,.375,.625,.875]).all()
    # A correct candidate with the fewest REAL tokens remains positive; longer correct candidates remain positive too.
    safe=np.argmin(np.where(Y==1,C,np.inf),1)
    failure=(Y==0)&(C<C[np.arange(len(C)),safe,None])
    negative=np.argmax(np.where(failure,C,-np.inf),1)
    valid=(Y.max(1)>0)&failure.any(1)
    unit=float(C[train,-1].mean());device='cuda'
    tx=torch.tensor(X,device=device);ty=torch.tensor(Y,device=device);tq=torch.tensor(Q,device=device);tc=torch.tensor(C,device=device)
    tb=torch.tensor(safe,device=device);tn=torch.tensor(negative,device=device);tv=torch.tensor(valid,device=device)
    tr=np.flatnonzero(train);dv=np.flatnonzero(dev);start=time.time()
    plan={'configs':CONFIGS,'seeds':list(SEEDS),'architecture':'2048-128-GELU-Dropout.15; correctness8/recovery8/reference logcost1/paired relative logcost7',
        'learning_rate':3e-4,'weight_decay':.01,'batch_size':128,'max_epochs':120,'early_stopping_patience':16,'minimum_epochs':24,
        'cost_loss_weight':.25,'relative_cost_loss_weight':.5,'boundary_margin_logits':.2,'policy_temperature':.1,'policy_cost_weight':.1,
        'train_n':6143,'dev_n':900,'target':'K3 Jeffreys smoothed recovery auxiliary plus actual one-pass correctness primary',
        'label_edits':False,'input_sha256':digest(ROOT/'runs/dev_v5_shared_joint_smooth/policy_predictions.jsonl'),'features_sha256':digest(ROOT/'data/shared_prefix_development_features.npz'),
        'source_sha256':{n:digest(ROOT/'src'/n) for n in ['sufficiency_model_v7.py','train_sufficiency_v7.py','select_sufficiency_v7.py']},'official_test_used':False}
    if (out/'PLAN.json').exists():assert json.loads((out/'PLAN.json').read_text())==plan,'Training plan changed; use a new version'
    else:dump(out/'PLAN.json',plan)
    for cfg in CONFIGS:
        dest=out/cfg['name'];dest.mkdir(exist_ok=True);np.savez_compressed(dest/'normalizer.npz',mean=mean,std=std)
        for seed in SEEDS:
            if (dest/f'predictions_seed{seed}.npz').exists():continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
            net=SufficiencyNet(X.shape[1]).to(device);opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
            best=float('inf');beststate=None;history=[];bestepoch=-1
            def lossfn(ids,training=True):
                s,q,lc,rel=net(tx[ids]);y=ty[ids];corrected=s+np.log(cfg['negative_weight'])
                safe_loss=(nn.functional.softplus(-s)*y+cfg['negative_weight']*nn.functional.softplus(s)*(1-y)).mean()
                soft=nn.functional.binary_cross_entropy_with_logits(q,tq[ids]);p=corrected.sigmoid()
                cost=nn.functional.smooth_l1_loss(lc,tc[ids].log())
                paired=nn.functional.smooth_l1_loss(rel,(tc[ids]/tc[ids,-1,None]).log())
                if not training:return nn.functional.binary_cross_entropy_with_logits(corrected,y)+.25*soft+.1*cost+.1*paired
                mono=(torch.relu(p[:,:-1]-p[:,1:]).mean()+torch.relu(q.sigmoid()[:,:-1]-q.sigmoid()[:,1:]).mean())/2
                mask=tv[ids];ar=torch.arange(len(ids),device=device)
                boundary=torch.relu(.2-s[ar,tb[ids]]+s[ar,tn[ids]])
                boundary=boundary[mask].mean() if mask.any() else s.sum()*0
                predicted_c=lc.detach().clamp(0,9).exp()/unit
                pi=torch.softmax((corrected-.1*predicted_c)/.1,-1)
                regret=(pi*((1-y)+.1*tc[ids]/unit)).sum(-1).mean()
                return safe_loss+cfg['soft_weight']*soft+.25*cost+.5*paired+cfg['mono_weight']*mono+cfg['boundary_weight']*boundary+cfg['policy_weight']*regret
            seed_start=time.time()
            for epoch in range(120):
                net.train();rng.shuffle(tr);losses=[]
                for offset in range(0,len(tr),128):
                    ids=tr[offset:offset+128];opt.zero_grad(set_to_none=True);loss=lossfn(ids);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss.detach()))
                net.eval()
                with torch.inference_mode():vl=float(lossfn(dv,False))
                history.append({'epoch':epoch,'train_objective':float(np.mean(losses)),'dev_unweighted_calibration_cost_loss':vl})
                if vl<best:
                    best=vl;bestepoch=epoch;beststate={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                if epoch>=24 and epoch-bestepoch>=16:break
            net.load_state_dict(beststate);net.eval()
            with torch.inference_mode():s,q,lc,_=net(tx);pp=(s+np.log(cfg['negative_weight'])).sigmoid().cpu().numpy();qq=q.sigmoid().cpu().numpy();logc=lc.cpu().numpy()
            # Train-only ratio-specific smearing corrects log-cost retransformation bias.
            smear=np.exp(np.log(C[train])-logc[train]).mean(0).clip(.25,4)
            pc=np.exp(logc.clip(0,9))*smear
            torch.save({'state_dict':beststate,'input_dim':X.shape[1],'seed':seed,'negative_weight':cfg['negative_weight'],'cost_smearing':smear.tolist(),'loss_config':cfg,'best_epoch':bestepoch},dest/f'model_seed{seed}.pt')
            np.savez_compressed(dest/f'predictions_seed{seed}.npz',ids=np.array([x['id'] for x in rows]),prob=pp,recovery=qq,cost=pc)
            dump(dest/f'history_seed{seed}.json',history)
            status={'status':'training','config':cfg['name'],'seed':seed,'epochs':len(history),'best_epoch':bestepoch,'best_dev_loss':best,'seed_wall_seconds':time.time()-seed_start,'total_wall_seconds':time.time()-start}
            dump(out/'status.json',status);print(json.dumps(status),flush=True)
        dump(dest/'COMPLETE.json',{'status':'trained','config':cfg,'seeds':list(SEEDS),'plan_sha256':digest(out/'PLAN.json')})
    dump(out/'TRAINING_COMPLETE.json',{'status':'complete','configs':len(CONFIGS),'models':len(CONFIGS)*len(SEEDS),'wall_seconds':time.time()-start})
if __name__=='__main__':main()
