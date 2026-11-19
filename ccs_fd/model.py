"""CCS-FD component prediction and local verification of the candidate state."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NUMERIC_FIELDS = ('class_mean', 'class_std', 'metric_mean', 'metric_std', 'cuts', 'thresholds')


def save_checkpoint(pack, path):
    """Store model weights and reference banks using PyTorch's restricted format."""
    portable = dict(pack)
    for field in NUMERIC_FIELDS:
        portable[field] = torch.as_tensor(portable[field]).detach().cpu()
    torch.save(portable, path)


class ComponentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backs = nn.ModuleList([
            nn.Sequential(nn.Linear(width, 192), nn.LayerNorm(192), nn.SiLU(),
                          nn.Linear(192, 96), nn.SiLU()) for width in (900, 600, 1128)])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 1)) for _ in range(5)])

    def forward(self, features):
        a, b, c = [block(features[:, lo:hi]) for block, lo, hi in
                   zip(self.backs, (0, 900, 1500), (900, 1500, 2628))]
        return torch.cat([head(value) for head, value in zip(self.heads, (a, b, c, c, c))], 1).sigmoid()


def make_encoder():
    return nn.Sequential(nn.Linear(1128, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 128))


def log_odds(probability):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


class Predictor:
    def __init__(self, checkpoint=None, device=None):
        checkpoint = checkpoint or Path(__file__).parent / 'checkpoints' / '1994217509.pt'
        self.device = str(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        p = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if p.get('format') != 'ccs-fd-simplified-v2':
            raise ValueError('Expected a CCS-FD simplified v2 checkpoint.')
        for field in NUMERIC_FIELDS:
            p[field] = p[field].numpy()
        self.pack = p
        self.classifier = ComponentClassifier().to(self.device).eval()
        self.classifier.load_state_dict(p['classifier'])
        self.encoder = make_encoder().to(self.device).eval()
        self.encoder.load_state_dict(p['encoder'])
        # Original float32 rows retained in the CPU checkpoint for rare exact rechecks.
        self.banks = [F.normalize(bank.double(), dim=1).float().to(self.device).contiguous()
                      for bank in p['banks']]
        self.labels = np.asarray(p['legal'])
        self.codes = np.array([[atom in label for atom in 'ABCDE'] for label in self.labels], dtype=float)
        self.offsets = log_odds(p['cuts'])
        self.last_exact_rechecks = 0

    @torch.inference_mode()
    def predict(self, features, return_detail=False):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 2628 or not np.isfinite(features).all():
            raise ValueError('Expected finite features with shape (N, 2628).')
        p = self.pack
        predictions, candidates, distances = [], [], []
        self.last_exact_rechecks = 0
        # Guard band is numerical, not a learned decision threshold. It changes
        # only whether the original float64 distance must be recomputed.
        reduced_precision = self.device.startswith('cuda') and torch.backends.cuda.matmul.allow_tf32
        error_guard = .01 if reduced_precision else 1e-4
        for start in range(0, len(features), 256):
            batch = features[start:start + 256]
            normalized = np.clip((batch - p['class_mean']) / p['class_std'], -12, 12)
            probability = self.classifier(torch.as_tensor(normalized, device=self.device)).cpu().numpy()
            candidate = self.labels[((log_odds(probability) - self.offsets) @ self.codes.T).argmax(1)]
            bearing = np.clip((batch[:, 1500:] - p['metric_mean']) / p['metric_std'], -12, 12)
            z = F.normalize(self.encoder(torch.as_tensor(bearing, device=self.device)), dim=1)
            unit = F.normalize(z.double(), dim=1)
            query = unit.float()
            hyp = np.array([''.join(c for c in name if c in 'CDE') or 'H' for name in candidate])
            accepted = np.zeros(len(batch), dtype=bool)
            ds = np.empty(len(batch)) if return_detail else None
            for j, name in enumerate(p['hypotheses']):
                ids = np.flatnonzero(hyp == name)
                if not len(ids):
                    continue
                limit = float(p['thresholds'][j])
                pair = (2 - 2 * query[ids] @ self.banks[j].T).clamp_min_(0)
                fifth = pair.topk(5, largest=False).values[:, -1].cpu().numpy().astype(np.float64)
                ambiguous = np.flatnonzero(np.abs(fifth - limit) <= error_guard)
                if len(ambiguous):
                    exact_bank = F.normalize(p['banks'][j].to(self.device).double(), dim=1)
                    # Keep the original 256-row query chunk for deterministic
                    # rounding at source quantiles that equal an observed distance.
                    exact_pair = (2 - 2 * unit @ exact_bank.T).clamp_min_(0)[ids[ambiguous]]
                    fifth[ambiguous] = exact_pair.topk(5, largest=False).values[:, -1].cpu().numpy()
                    self.last_exact_rechecks += len(ambiguous)
                accepted[ids] = fifth <= limit
                if return_detail:
                    ds[ids] = fifth
            predictions.append(np.where(accepted, candidate, 'UNKNOWN'))
            if return_detail:
                candidates.append(candidate)
                distances.append(ds)
        pred = np.concatenate(predictions) if predictions else np.empty(0, dtype='<U7')
        if return_detail:
            return dict(pred=pred, label=np.concatenate(candidates) if candidates else np.empty(0, dtype='<U2'),
                        distance=np.concatenate(distances) if distances else np.empty(0),
                        exact_rechecks=self.last_exact_rechecks)
        return pred

    def predict_windows(self, windows, return_detail=False):
        from .features import extract_features
        return self.predict(extract_features(windows, self.device), return_detail)
