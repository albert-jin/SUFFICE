"""Deployable query-feature controller. No answers or candidate outputs accepted."""
import torch
from torch import nn
from common import *
class ControllerNet(nn.Module):
    def __init__(self,dim,form):
        super().__init__();self.form=form;self.trunk=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Dropout(.15));self.joint=nn.Linear(128,32);self.recover=nn.Linear(128,8);self.cost=nn.Linear(128,8);self.reference=nn.Linear(128,1);self.conditional=nn.Linear(128,16);self.marginal=nn.Linear(128,8)
    def forward(self,x):
        h=self.trunk(x)
        if self.form=='joint':logits=self.joint(h).reshape(-1,8,4)
        else:
            if self.form=='conditional':
                ref=self.reference(h).sigmoid();cond=self.conditional(h).reshape(-1,8,2).sigmoid();p0=cond[:,:,0];p1=cond[:,:,1]
            else:
                m=self.marginal(h).sigmoid();ref=m[:,-1:];p0=m;p1=m
            joint=torch.stack([(1-ref)*(1-p0),(1-ref)*p0,ref*(1-p1),ref*p1],dim=-1);rr=ref[:,0];zero=torch.zeros_like(rr);reference=torch.stack([1-rr,zero,zero,rr],dim=-1)[:,None,:]
            logits=torch.cat([joint[:,:-1,:],reference],dim=1).clamp_min(1e-7).log()
        return logits,self.recover(h),self.cost(h)
class Controller:
    def __init__(self,path):
        torch.set_num_threads(4);self.path=Path(path);self.summary=json.loads((self.path/'COMPLETE.json').read_text());norm=np.load(self.path/'normalizer.npz');self.mean=norm['mean'];self.std=norm['std'];self.models=[]
        for seed in SEEDS:
            meta=torch.load(self.path/f'model_seed{seed}.pt',map_location='cpu',weights_only=True);net=ControllerNet(meta['input_dim'],meta['form']);net.load_state_dict(meta['state_dict'],strict=True);net.eval();self.models.append((net,np.asarray(meta['cost_smearing'],dtype=np.float32)))
    def predict(self,features):
        x=torch.tensor((np.asarray(features,dtype=np.float32)-self.mean)/self.std);joints=[];costs=[];recover=[]
        with torch.inference_mode():
            for net,smear in self.models:
                j,q,c=net(x);joints.append(j.softmax(-1).numpy());recover.append(q.sigmoid().numpy());costs.append((np.exp(c.numpy().clip(0,9))*smear-1).clip(1))
        joints=np.stack(joints);p=joints.mean(0);cost=np.mean(costs,axis=0);unc=(joints[...,2]-joints[...,1]).std(0);choice=self.summary['choice']
        action=select_actions(p,cost,choice['threshold'],unc,choice['beta']) if choice['threshold'] is not None else np.full(len(features),7)
        return {'actions':action,'joint':p,'cost':cost,'recoverability':np.mean(recover,axis=0),'seed_joints':joints,'seed_costs':np.array(costs)}

