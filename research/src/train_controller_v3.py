"""Paired harm/rescue controller with recoverability supervision and actual cost.

All fitting and policy selection use train/dev only. Calibration is an audit,
not an additional tuning set. Final test access is deliberately absent here.
"""
import argparse,time,copy,json
from common import *
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--features',required=True);ap.add_argument('--retention',required=True);ap.add_argument('--recovery',required=True);ap.add_argument('--out',required=True);ap.add_argument('--label',choices=['smooth','raw','majority','single','none'],default='smooth');ap.add_argument('--mono',type=float,default=0);ap.add_argument('--epochs',type=int,default=160)
    ap.add_argument('--form',choices=['joint','conditional','marginal'],default='conditional')
    args=ap.parse_args()
    import torch
    from torch import nn
    torch.set_num_threads(4);start=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():print('ALREADY_COMPLETE');return
    rows=readl(args.input);assert all(x['split'] in ['train','dev','calibration','reserve'] for x in rows)
    feat=np.load(args.features);fmap={k:v for k,v in zip(feat['ids'],feat['features'])}
    retained=readl(args.retention)
    assert all(x['scoring_error'] is None and x['correct'] in (0,1) for x in retained),'Unresolved grading errors'
    ret={(x['id'],x['ratio']):x for x in retained};assert len(ret)==len(retained),'Duplicate retained candidates'
    rec={}
    for x in readl(args.recovery):
        assert x['scoring_error'] is None and x['correct'] in (0,1) and x['condition']=='true','Invalid recovery label'
        rec.setdefault((x['id'],x['ratio']),[]).append(x)
    assert all(len(v)==3 and len({x['seed'] for x in v})==3 and len({x['cot_sha256'] for x in v})==1 for v in rec.values())
    examples=[];xs=[];ys=[];costs=[];recover=[]
    for row in rows:
        if not all((row['id'],r) in ret and len(rec.get((row['id'],r),[]))==3 for r in RATIOS):continue
        examples.append(row);xs.append(fmap[row['id']]);ys.append([ret[(row['id'],r)]['correct'] for r in RATIOS]);costs.append([ret[(row['id'],r)]['output_tokens'] for r in RATIOS]);recover.append([[x['correct'] for x in sorted(rec[(row['id'],r)],key=lambda x:x['seed'])] for r in RATIOS])
    X=np.array(xs,dtype=np.float32);Y=np.array(ys,dtype=np.int64);C=np.array(costs,dtype=np.float32);Q=np.array(recover,dtype=np.float32);J=paired_targets(Y)
    train=np.array([x['split']=='train' for x in examples]);dev=np.array([x['split']=='dev' for x in examples]);assert train.sum()>16 and dev.sum()>16
    mean=X[train].mean(0);std=X[train].std(0).clip(.05);X=(X-mean)/std
    np.savez_compressed(out/'normalizer.npz',mean=mean,std=std)
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.trunk=nn.Sequential(nn.Linear(X.shape[1],128),nn.GELU(),nn.Dropout(.15));self.joint=nn.Linear(128,32);self.recover=nn.Linear(128,8);self.cost=nn.Linear(128,8)
            self.reference=nn.Linear(128,1);self.conditional=nn.Linear(128,16);self.marginal=nn.Linear(128,8)
            nn.init.zeros_(self.cost.weight)
            with torch.no_grad():self.cost.bias.copy_(torch.tensor(np.log1p(C[train]).mean(0)))
        def forward(self,x):
            h=self.trunk(x)
            if args.form=='joint':logits=self.joint(h).reshape(-1,8,4)
            else:
                if args.form=='conditional':
                    ref=self.reference(h).sigmoid();cond=self.conditional(h).reshape(-1,8,2).sigmoid();p0=cond[:,:,0];p1=cond[:,:,1]
                else:
                    m=self.marginal(h).sigmoid();ref=m[:,-1:];p0=m;p1=m
                joint=torch.stack([(1-ref)*(1-p0),(1-ref)*p0,ref*(1-p1),ref*p1],dim=-1)
                rr=ref[:,0];zero=torch.zeros_like(rr);reference=torch.stack([1-rr,zero,zero,rr],dim=-1)[:,None,:]
                joint=torch.cat([joint[:,:-1,:],reference],dim=1);logits=joint.clamp_min(1e-7).log()
            return logits,self.recover(h),self.cost(h)
    tx=torch.tensor(X);tj=torch.tensor(J);tc=torch.tensor(np.log1p(C));ty=torch.tensor(Y,dtype=torch.float32);preds=[];metrics=[]
    for seed in SEEDS:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=Net();opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
        counts=Q.sum(-1)
        if args.label=='smooth':target=(counts+.5)/4
        elif args.label=='raw':target=counts/3
        elif args.label=='majority':target=(counts>=2).astype(float)
        elif args.label=='single':
            choices=rng.integers(3,size=counts.shape);target=np.take_along_axis(Q,choices[...,None],axis=-1).squeeze(-1)
        else:target=np.zeros_like(counts)
        tq=torch.tensor(target,dtype=torch.float32)
        def objective(ids):
            joint,qr,c=net(tx[ids])
            if args.form=='marginal':
                jp=joint.softmax(-1);loss=nn.functional.binary_cross_entropy(jp[:,:,1]+jp[:,:,3],ty[ids])
            else:loss=nn.functional.cross_entropy(joint.reshape(-1,4),tj[ids].reshape(-1))
            loss=loss+.1*nn.functional.smooth_l1_loss(c,tc[ids])
            if args.label!='none':loss=loss+.25*nn.functional.binary_cross_entropy_with_logits(qr,tq[ids])
            if args.mono:loss=loss+args.mono*nn.functional.relu(qr.sigmoid()[:,:-1]-qr.sigmoid()[:,1:]).mean()
            return loss
        best=float('inf');state=None;history=[];tr=np.where(train)[0];dv=np.where(dev)[0]
        for epoch in range(args.epochs):
            net.train();rng.shuffle(tr);losses=[]
            for s in range(0,len(tr),64):
                opt.zero_grad();loss=objective(tr[s:s+64]);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
            net.eval()
            with torch.no_grad():vl=float(objective(dv))
            history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'dev_loss':vl})
            if vl<best:best=vl;state=copy.deepcopy(net.state_dict());best_epoch=epoch
        net.load_state_dict(state);net.eval()
        with torch.no_grad():j,q,c=net(tx);p=j.softmax(-1).numpy();q=q.sigmoid().numpy();rawcost=np.exp(c.numpy().clip(0,9))
        smearing=((C[train]+1)/rawcost[train]).mean(0);cost=(rawcost*smearing-1).clip(1)
        assert np.isfinite(p).all() and np.isfinite(cost).all()
        if args.form!='joint':assert np.ptp(p[:,:,2]+p[:,:,3],axis=1).max()<1e-5,'Reference probability must be shared across ratios'
        torch.save({'state_dict':state,'input_dim':X.shape[1],'seed':seed,'label':args.label,'form':args.form,'cost_smearing':smearing.tolist(),'mono':args.mono,'architecture':'Shared128; consistent reference/conditional rescue-preserve OR joint OR marginal; recovery8/logcost8'},out/f'model_seed{seed}.pt')
        np.savez_compressed(out/f'predictions_seed{seed}.npz',ids=np.array([x['id'] for x in examples]),joint=p,recovery=q,cost=cost)
        dump(out/f'history_seed{seed}.json',history);preds.append((p,q,cost));metrics.append({'seed':seed,'best_epoch':best_epoch,'best_dev_loss':best,'dev_cost_relative_mae':float(np.abs(cost[dev]-C[dev]).mean()/C[dev].mean()),'dev_cost_prediction_mean':float(cost[dev].mean()),'reference_marginal_max_span':float(np.ptp(p[:,:,2]+p[:,:,3],axis=1).max())})
    jp=np.stack([z[0] for z in preds]);q=np.mean([z[1] for z in preds],axis=0);cost=np.mean([z[2] for z in preds],axis=0);p=jp.mean(0);unc=(jp[...,2]-jp[...,1]).std(0)
    policies=[];index=np.arange(len(examples));tau_grid=[-.10,-.05,-.02,0,.01,.02,.04,.06,.08,.10,.15,.20]
    def summary(action,mask):
        yy=Y[index,action];cc=C[index,action]
        ds={d:float(yy[mask & np.array([x['dataset']==d for x in examples])].mean()) for d in DATASETS if (mask & np.array([x['dataset']==d for x in examples])).any()}
        return {'n':int(mask.sum()),'accuracy':float(yy[mask].mean()),'mean_output_tokens':float(cc[mask].mean()),'token_saving_vs_r1':float(1-cc[mask].mean()/C[mask,-1].mean()),'delta_accuracy_vs_r1':float((yy-Y[:,-1])[mask].mean()),'retention_distribution':{str(r):float((action[mask]==j).mean()) for j,r in enumerate(RATIOS)},'per_domain_accuracy':ds}
    for beta in [0,.5,1.]:
        for tau in tau_grid:
            action=select_actions(p,cost,tau,unc,beta);s=summary(action,dev);policies.append({'beta':beta,'threshold':tau,'dev':s})
    feasible=[v for v in policies if v['dev']['delta_accuracy_vs_r1']>=-.01 and v['dev']['token_saving_vs_r1']>0]
    choice=min(feasible,key=lambda x:(x['dev']['mean_output_tokens'],-x['dev']['accuracy'],x['beta'])) if feasible else {'beta':0.,'threshold':None,'fallback':'all_full_retention'}
    action=select_actions(p,cost,choice['threshold'],unc,choice['beta']) if choice['threshold'] is not None else np.full(len(examples),7)
    records=[]
    for i,x in enumerate(examples):
        records.append({'id':x['id'],'dataset':x['dataset'],'split':x['split'],'selected_ratio':RATIOS[action[i]],'selected_index':int(action[i]),'correct':int(Y[i,action[i]]),'output_tokens':int(C[i,action[i]]),'reference_correct':int(Y[i,-1]),'reference_output_tokens':int(C[i,-1]),'predicted_joint':p[i].tolist(),'predicted_recoverability':q[i].tolist(),'actual_recoverability':((Q[i].sum(-1)+.5)/4).tolist(),'candidate_correctness':Y[i].tolist(),'candidate_tokens':C[i].tolist()})
    writel(out/'policy_predictions.jsonl',records)
    fixed={str(r):summary(np.full(len(examples),j),dev) for j,r in enumerate(RATIOS)}
    result={'name':'paired_harm_rescue_cost_v3','form':args.form,'cost_initialization':'train log-cost mean; zero residual weights; train-only smearing correction','status':'development_only','label':args.label,'mono':args.mono,'trained_seeds':list(SEEDS),'choice':choice,'fixed_dev':fixed,'train':summary(action,train),'dev':summary(action,dev),'calibration':summary(action,np.array([x['split']=='calibration' for x in examples])) if any(x['split']=='calibration' for x in examples) else None,'fit_metrics':metrics,'n_complete_rows':len(examples),'input_n':len(rows),'wall_seconds':time.time()-start,'warning':'Pilot development replay is not final-test evidence, nor measured online latency. Ensemble deployment cost and same-model feature prefill must be included.'}
    dump(out/'policy_search.json',policies);dump(out/'COMPLETE.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
