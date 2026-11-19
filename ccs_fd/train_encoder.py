"""Standalone support-encoder training, without state-discrimination or reconstruction losses.

train(data, seed, device) accepts the existing source data dictionary:
x (N,2628), names, augmentation, context (0..8), recipe (N,4).
Only real source A-E and source-synthetic AB/CD are allowed.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .model import make_encoder

HYPOTHESES = ('H', 'C', 'D', 'E', 'CD')


def make_episodes(data, seed):
    recipe = data['recipe']
    left = (recipe[:, 1] < 54) & ((recipe[:, 3] < 0) | (recipe[:, 3] < 54))
    right = (recipe[:, 1] >= 54) & ((recipe[:, 3] < 0) | (recipe[:, 3] >= 54))
    hypothesis = np.array([''.join(a for a in name if a in 'CDE') or 'H' for name in data['names']])
    rng = np.random.default_rng(seed)
    pools = {}

    def sample(mask, count, key):
        if key not in pools:
            pools[key] = [[np.flatnonzero(mask & (data['context'] == context) & (data['names'] == name))
                           for name in np.unique(data['names'][mask & (data['context'] == context)])]
                          for context in np.unique(data['context'][mask])]
        groups = pools[key]
        chosen = rng.integers(len(groups), size=count)
        out = []
        for index in chosen:
            rows = groups[index][rng.integers(len(groups[index]))]
            out.append(int(rows[rng.integers(len(rows))]))
        return np.array(out)

    draws = []
    rotation = 0
    for step in range(40 * 32):
        held = '' if step % 4 == 0 else 'CDE'[rotation % 3]
        if held:
            rotation += 1
        load = -1
        rmask = left
        qmask = right
        names = [h for h in HYPOTHESES if (not held or held not in h) and (hypothesis == h).any()]
        refs = np.concatenate([sample(rmask & (hypothesis == h), 64, ('r', load, h)) for h in names])
        queries = np.concatenate([sample(qmask & (hypothesis == h), 16, ('q', load, h)) for h in names]
            + ([sample(qmask & (data['names'] == held), 64, ('u', load, held))] if held else []))
        draws.append(dict(refs=refs, queries=queries, names=names))
    return draws


def hard_average(values):
    return .5 * values.mean() + .5 * values.topk(max(1, (len(values) + 3) // 4)).values.mean()


def train(data, seed, device='cuda'):
    if not set(data['names']).issubset({'A', 'B', 'C', 'D', 'E', 'AB', 'CD'}):
        raise ValueError('Use source A-E and source-synthetic AB/CD only.')
    pure = (np.char.str_len(data['names']) == 1) & (data['augmentation'] == 'clean')
    mean = data['x'][pure].mean(0)[1500:]
    std = np.maximum(data['x'][pure].std(0), .05)[1500:]
    features = torch.as_tensor(np.clip((data['x'][:, 1500:] - mean) / std, -12, 12), device=device)
    draws = make_episodes(data, seed)
    torch.manual_seed(seed)
    model = make_encoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, eta_min=1e-5)
    history = []
    for epoch in range(40):
        total = 0.
        for ep in draws[epoch * 32:(epoch + 1) * 32]:
            nr, nc = len(ep['refs']), len(ep['names'])
            batch = features[torch.as_tensor(np.r_[ep['refs'], ep['queries']], device=device)]
            z = F.normalize(model(batch), dim=1)
            refs = z[:nr].reshape(nc, 64, 128)
            distances = (2 - 2 * torch.einsum('nd,kmd->nkm', z[nr:], refs)).clamp_min(0).topk(5, dim=2, largest=False).values[:, :, -1]
            nk = 16 * nc
            target = torch.arange(nc, device=device).repeat_interleave(16)
            positive = distances[torch.arange(nk, device=device), target]
            loss = hard_average(F.softplus((positive - .2) / .1))
            if len(distances) > nk:
                loss = loss + hard_average(F.softplus((.6 - distances[nk:].min(1).values) / .1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        history.append(total / 32)
    return model.eval(), mean, std, history
