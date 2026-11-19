"""Train the component predictor and export its five inference heads."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TrainingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backs = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 192), nn.LayerNorm(192), nn.SiLU(),
                          nn.Linear(192, 96), nn.SiLU())
            for width in (900, 600, 1128)
        ])
        # A, B, no bearing fault, C, D, E. The auxiliary output participates
        # in fitting only. Module creation order preserves the paper's seeds.
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 32))
            for _ in range(6)
        ])
        self.head = nn.ModuleList([nn.Linear(32, 1) for _ in range(6)])

    def forward(self, x):
        motor = self.backs[0](x[:, :900])
        gear = self.backs[1](x[:, 900:1500])
        bearing = self.backs[2](x[:, 1500:])
        branches = (motor, gear, bearing, bearing, bearing, bearing)
        z = torch.stack([layer(branch) for layer, branch in zip(self.proj, branches)], 1)
        return torch.cat([head(z[:, i]) for i, head in enumerate(self.head)], 1)

    def inference_weights(self):
        state = self.cpu().state_dict()
        weights = {k: v for k, v in state.items() if k.startswith('backs.')}
        for new, old in enumerate((0, 1, 3, 4, 5)):
            source, dest = f'proj.{old}', f'heads.{new}'
            output = state[f'head.{old}.weight'].double()
            weights[dest + '.0.weight'] = state[source + '.0.weight']
            weights[dest + '.0.bias'] = state[source + '.0.bias']
            weights[dest + '.2.weight'] = (output @ state[source + '.2.weight'].double()).float()
            weights[dest + '.2.bias'] = (
                output @ state[source + '.2.bias'].double() + state[f'head.{old}.bias'].double()
            ).float()
        return weights


def balanced_batches(data, seed):
    keys = np.array([f'{c}/{g}/{a}' for c, g, a in
                     zip(data['context'], data['groups'], data['augmentation'])])
    pools = [np.flatnonzero(keys == key) for key in np.unique(keys)]
    sizes = np.array([len(pool) for pool in pools])
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    rng = np.random.default_rng(seed)
    selected = rng.integers(len(pools), size=(50, 48, 256))
    rows = (rng.random(selected.shape) * sizes[selected]).astype(int)
    return np.concatenate(pools)[offsets[selected] + rows]


def balanced_loss(logits, targets):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    positives = targets.sum(0)
    negatives = (1 - targets).sum(0)
    pos_loss = (ce * targets).sum(0) / positives.clamp_min(1)
    neg_loss = (ce * (1 - targets)).sum(0) / negatives.clamp_min(1)
    return ((pos_loss + neg_loss) / (positives.gt(0).float() + negatives.gt(0).float())).mean()


def evidence_cutoffs(scores, names):
    labels = np.unique(names)
    cutoffs = []
    for j, atom in enumerate('ABCDE'):
        best = (-1., -1.)
        for threshold in np.linspace(.01, .99, 99):
            present, absent = [], []
            for label in labels:
                hits = scores[names == label, j] > threshold
                if atom in label:
                    present.append(hits.mean())
                else:
                    absent.append((~hits).mean())
            best = max(best, (.5 * (np.mean(present) + np.mean(absent)), threshold))
        cutoffs.append(best[1])
    return np.asarray(cutoffs)


def fit(train, calibration, seed, device):
    pure = (np.char.str_len(train['names']) == 1) & (train['augmentation'] == 'clean')
    singles = train['x'][pure]
    mean, std = singles.mean(0), np.maximum(singles.std(0), .05)
    x = torch.as_tensor((train['x'] - mean) / std, device=device).clamp(-12, 12)
    codes = np.array([[a in name for a in 'ABCDE'] for name in train['names']], dtype=np.float32)
    targets = np.c_[codes[:, :2], (codes[:, 2:].sum(1) == 0).astype(np.float32), codes[:, 2:]]
    y = torch.as_tensor(targets, device=device)
    torch.manual_seed(seed)
    model = TrainingClassifier().to(device)
    batches = torch.as_tensor(balanced_batches(train, seed), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(batches), eta_min=1e-5)
    for epoch in batches:
        for rows in epoch:
            loss = balanced_loss(model(x[rows]), y[rows])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        schedule.step()
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor((calibration['x'] - mean) / std, device=device).clamp(-12, 12)
        scores = torch.cat([model(batch)[:, [0, 1, 3, 4, 5]].sigmoid() for batch in x.split(256)])
        cutoffs = evidence_cutoffs(scores.cpu().numpy(), calibration['names'])
    return model.inference_weights(), mean, std, cutoffs
