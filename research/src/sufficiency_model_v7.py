"""MD-inspired calibrated sufficiency and paired real-cost prediction."""
import torch
from torch import nn
from common import *

class SufficiencyNet(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.Dropout(.15))
        self.safe=nn.Linear(128,8);self.soft=nn.Linear(128,8)
        self.reference_cost=nn.Linear(128,1);self.relative_cost=nn.Linear(128,7)
    def forward(self,x):
        h=self.trunk(x);s=self.safe(h);q=self.soft(h)
        ref=self.reference_cost(h);relative=torch.cat([self.relative_cost(h),torch.zeros_like(ref)],-1)
        logcost=ref+relative
        return s,q,logcost,relative

def choose_sufficient(prob,cost,config,uncertainty=None):
    p=np.asarray(prob)
    if uncertainty is not None:p=p-config['beta']*np.asarray(uncertainty)
    threshold=np.maximum(config['tau'],p[:,-1]-config['reference_slack'])
    allowed=(p>=threshold[:,None])&(np.asarray(RATIOS)[None,:]>=config['min_ratio'])
    allowed[:,-1]=True
    return np.argmin(np.where(allowed,cost,np.inf),axis=1)

class SufficiencyRouter:
    def __init__(self,path,config):
        torch.set_num_threads(4);path=Path(path);self.config=config;self.models=[]
        norm=np.load(path/'normalizer.npz');self.mean=norm['mean'];self.std=norm['std'];self.last=[]
        for seed in SEEDS:
            meta=torch.load(path/f'model_seed{seed}.pt',map_location='cpu',weights_only=True)
            net=SufficiencyNet(meta['input_dim']);net.load_state_dict(meta['state_dict']);net.eval()
            self.models.append((net,meta['negative_weight'],np.asarray(meta['cost_smearing'])))
    def route(self,features):
        x=torch.tensor((np.asarray(features,np.float32)-self.mean)/self.std);pp=[];cc=[];qq=[]
        with torch.inference_mode():
            for net,w,smear in self.models:
                s,q,lc,_=net(x)
                # Weighted BCE logits are NOT calibrated probabilities: undo class weighting.
                pp.append((s+np.log(w)).sigmoid().numpy());qq.append(q.sigmoid().numpy())
                cc.append(np.exp(lc.numpy().clip(0,9))*smear)
        p=np.mean(pp,0);c=np.mean(cc,0);q=np.mean(qq,0);u=np.std(pp,0)
        a=choose_sufficient(p,c,self.config,u)
        self.last=[{'predicted_correct_probability':float(p[i,j]),'predicted_soft_recoverability':float(q[i,j]),'predicted_output_tokens':float(c[i,j]),'action_index':int(j)} for i,j in enumerate(a)]
        return a
