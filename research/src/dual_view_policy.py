"""Two-view decision rule. Inputs are model predictions, never ground truth."""
import numpy as np
from common import RATIOS
def choose_dual_view(seed_joint,seed_recovery,seed_cost,*,risk_threshold,recoverability_loss_limit,beta,min_ratio=.5):
    j=np.asarray(seed_joint);q=np.asarray(seed_recovery);c=np.asarray(seed_cost)
    assert j.ndim==4 and j.shape[-2:]==(8,4) and q.shape==c.shape==j.shape[:-1]
    task_loss=j[...,2]-j[...,1]
    sufficient_loss=q[:,:,-1,None]-q
    task_score=task_loss.mean(0)+beta*task_loss.std(0)
    sufficient_score=sufficient_loss.mean(0)+beta*sufficient_loss.std(0)
    allowed=(task_score<=risk_threshold)&(sufficient_score<=recoverability_loss_limit)&(np.array(RATIOS)[None,:]>=min_ratio)
    allowed[:,-1]=True
    return np.argmin(np.where(allowed,c.mean(0),np.inf),axis=1)
