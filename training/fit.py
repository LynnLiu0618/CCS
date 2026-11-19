"""Fit both CCS-FD networks and calibrate a portable inference checkpoint."""

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.nn import functional as F

from ccs_fd.calibration import HYPOTHESES, fit_limits
from ccs_fd.model import save_checkpoint
from ccs_fd.train_encoder import train as fit_encoder
from .components import fit as fit_components


LABELS = ['A', 'B', 'C', 'D', 'E', 'AB', 'CD']


def configure_training():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_table(path):
    with np.load(path, allow_pickle=False) as archive:
        keys = ('x', 'names', 'context', 'groups', 'augmentation', 'recipe')
        data = {key: archive[key] for key in keys}
    x = data['x']
    if x.ndim != 2 or x.shape[1] != 2628 or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f'{path}: expected finite float32 features of shape (N, 2628).')
    for field in ('names', 'context', 'groups', 'augmentation'):
        if data[field].shape != (len(x),):
            raise ValueError(f'{path}: {field} must contain one value per feature row.')
    if not np.isin(data['context'], range(9)).all():
        raise ValueError(f'{path}: invalid operating condition index.')
    if set(np.unique(data['names'])) != set(LABELS):
        raise ValueError(f'{path}: expected all five source singletons and synthetic AB/CD.')
    if set(np.unique(data['augmentation'])) != {'both', 'clean'}:
        raise ValueError(f'{path}: expected the original clean and augmented views.')
    recipe = data['recipe']
    if recipe.ndim != 2 or recipe.shape[0] != len(x) or recipe.shape[1] < 4:
        raise ValueError(f'{path}: expected at least four recipe columns per feature row.')
    if not np.isfinite(recipe[:, :4]).all():
        raise ValueError(f'{path}: invalid parent identifiers.')
    singles = {label: idx for idx, label in enumerate('ABCDE')}
    for label in LABELS:
        rows = recipe[data['names'] == label, :4]
        parents = sorted(singles[c] for c in label)
        if len(parents) == 1:
            valid = (rows[:, 0] == parents[0]) & (rows[:, 2] < 0) & (rows[:, 3] < 0)
        else:
            valid = np.all(np.sort(rows[:, [0, 2]], axis=1) == parents, axis=1) & (rows[:, 3] >= 0)
        if not (valid & (rows[:, 1] >= 0)).all():
            raise ValueError(f'{path}: {label} has inconsistent source parent identifiers.')
    return data


def clean_rows(data):
    mask = data['augmentation'] == 'clean'
    result = {key: values[mask] for key, values in data.items()}
    if set(np.unique(result['names'])) != set(LABELS):
        raise ValueError('The clean source view is missing a fault label.')
    return result


@torch.inference_mode()
def embed(encoder, features, mean, std, device):
    x = np.clip((features[:, 1500:] - mean) / std, -12, 12)
    return torch.cat([F.normalize(encoder(batch), dim=1)
                      for batch in torch.as_tensor(x, device=device).split(256)])


@torch.inference_mode()
def bank_distances(queries, banks):
    queries = F.normalize(queries.double(), dim=1)
    columns = []
    for bank in banks:
        unit = F.normalize(bank.double(), dim=1)
        distances = []
        for batch in queries.split(256):
            pair = (2 - 2 * batch @ unit.T).clamp_min(0)
            distances.append(pair.topk(5, largest=False).values[:, -1])
        columns.append(torch.cat(distances))
    return torch.stack(columns, dim=1).cpu().numpy()


def fit(train, calibration, paired_calibration, seed, device):
    clean_train = clean_rows(train)
    clean_cal = clean_rows(calibration)
    if (paired_calibration.shape != clean_cal['x'].shape or paired_calibration.dtype != np.float32
            or not np.isfinite(paired_calibration).all()):
        raise ValueError('The paired verifier calibration features must align with clean calibration rows.')
    labels = clean_train['names']
    recipe = clean_train['recipe']
    left = (recipe[:, 1] < 54) & (recipe[:, 3] < 54)
    right = (recipe[:, 1] >= 54) & ((recipe[:, 3] < 0) | (recipe[:, 3] >= 54))
    for mask in (left, right):
        if set(np.unique(labels[mask])) != set(LABELS):
            raise ValueError('Both source parent-window pools must cover all seven diagnostic labels.')
    print('Fitting component predictor', flush=True)
    classifier, class_mean, class_std, cuts = fit_components(train, calibration, seed, device)
    print('Fitting support encoder', flush=True)
    encoder, mean, std, _ = fit_encoder(clean_train, seed, device)
    ref = embed(encoder, clean_train['x'], mean, std, device)
    states = np.array([''.join(atom for atom in name if atom in 'CDE') or 'H' for name in labels])
    banks = [ref[states == state] for state in HYPOTHESES]
    if any(len(bank) < 5 for bank in banks):
        raise ValueError('Each reference state needs at least five clean source observations.')
    print('Calibrating state limits', flush=True)
    clean, augmented = [
        bank_distances(embed(encoder, features, mean, std, device), banks)
        for features in (clean_cal['x'], paired_calibration)
    ]
    return dict(format='ccs-fd-simplified-v2', seed=seed, classifier=classifier,
                encoder=encoder.cpu().state_dict(), class_mean=class_mean, class_std=class_std,
                metric_mean=mean, metric_std=std, cuts=cuts, legal=LABELS, hypotheses=HYPOTHESES,
                banks=[bank.cpu() for bank in banks],
                thresholds=fit_limits(clean, augmented, clean_cal['names']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Directory of prepared source features')
    parser.add_argument('--output', type=Path, required=True, help='Destination .pt checkpoint')
    parser.add_argument('--seed', type=int, default=1994217509)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    configure_training()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train = load_table(args.data / 'train.npz')
    calibration = load_table(args.data / 'calibration.npz')
    augmented = np.load(args.data / 'calibration_augmented.npy', allow_pickle=False)
    pack = fit(train, calibration, augmented, args.seed, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix='.pt', delete=False) as handle:
        temp = Path(handle.name)
    try:
        save_checkpoint(pack, temp)
        temp.replace(args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
