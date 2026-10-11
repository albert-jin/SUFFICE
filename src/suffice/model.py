"""Checkpoint-compatible three-head probe; only sufficiency/cost control routing."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

class SufficiencyNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(.15))
        self.safe = nn.Linear(128, 8)
        self.soft = nn.Linear(128, 8)
        self.reference_cost = nn.Linear(128, 1)
        self.relative_cost = nn.Linear(128, 7)

    def forward(self, x):
        hidden = self.trunk(x)
        reference = self.reference_cost(hidden)
        relative = torch.cat([self.relative_cost(hidden), torch.zeros_like(reference)], dim=-1)
        return self.safe(hidden), self.soft(hidden), reference + relative, relative

class ProbeEnsemble:
    def __init__(self, directory, seeds=(17, 43, 79)):
        directory = Path(directory)
        with np.load(directory / 'normalizer.npz', allow_pickle=False) as norm:
            self.mean = norm['mean'].astype(np.float32)
            self.std = norm['std'].astype(np.float32)
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Invalid feature normalizer.')
        if not seeds:
            raise ValueError('An ensemble must contain at least one probe.')
        self.models = []
        for seed in seeds:
            meta = torch.load(directory / f'model_seed{seed}.pt', map_location='cpu', weights_only=True)
            if meta['input_dim'] != len(self.mean):
                raise ValueError('Checkpoint and feature normalizer dimensions differ.')
            weight = float(meta['negative_weight'])
            smearing = np.asarray(meta['cost_smearing'], dtype=float)
            if not np.isfinite(weight) or weight <= 0 or smearing.shape != (8,) or not np.isfinite(smearing).all() or np.any(smearing <= 0):
                raise ValueError('Invalid calibration metadata.')
            net = SufficiencyNet(meta['input_dim'])
            net.load_state_dict(meta['state_dict'], strict=True)
            self.models.append((net.eval(), weight, smearing))

    def predict(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != len(self.mean) or not np.isfinite(features).all():
            raise ValueError('Features must be finite with the checkpoint input dimension.')
        x = torch.from_numpy((features - self.mean) / self.std)
        probabilities, costs, recovery = [], [], []
        with torch.inference_mode():
            for net, weight, smearing in self.models:
                logits, auxiliary, log_cost, _ = net(x)
                probabilities.append((logits + np.log(weight)).sigmoid().numpy())
                costs.append(np.exp(log_cost.numpy().clip(0, 9)) * smearing)
                recovery.append(auxiliary.sigmoid().numpy())
        return {'prob': np.mean(probabilities, axis=0), 'cost': np.mean(costs, axis=0),
                'recoverability_diagnostic': np.mean(recovery, axis=0),
                'prob_std': np.std(probabilities, axis=0)}
