"""Reconstruct source mixtures, classifier views and paired verification views."""

import numpy as np
import torch


ATOMS = "ABCDE"
RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)


def recipes(part, context):
    """Recreate the fixed parent draws, in the order used for the paper."""
    train = part == "train"
    rng = np.random.default_rng((87000 if train else 88000) + context)
    windows = np.arange(109 if train else 31)
    rows, names, groups, families = [], [], [], []

    def append(label, group, family, row):
        names.append(label)
        groups.append(group)
        families.append(family)
        rows.append(row)

    for index, label in enumerate(ATOMS):
        for parent in windows:
            append(label, label, "pure", [index, parent, -1, -1, 0, 0, *np.ones(6)])

    definitions = []
    for label in ("AB", "CD", "CE", "DE"):
        families_for_pair = ("axle",)
        if label == "AB":
            families_for_pair = ("propagate",) if train else ("block", "propagate")
        for family in families_for_pair:
            definitions.extend((label, family, ratio) for ratio in RATIOS)
    if not train:
        for label in ("AB", "CD"):
            definitions.extend((label, "stress", ratio) for ratio in (0.3, 0.6, 1.2, 2.4))

    for label, family, ratio in definitions:
        width = np.log(1.5 if family == "stress" else 1.25)
        for _ in range(64 if train else 31):
            left, right = rng.choice(windows, 2, replace=True)
            sign = rng.choice([-1, 1])
            gains = np.exp(rng.uniform(-width, width, 6))
            # Historical recipe generator also drew then discarded CE/DE.
            # Consuming these draws preserves subsequent calibration stress mixtures.
            if label not in ("AB", "CD"):
                continue
            if family not in ("propagate", "stress"):
                gains = np.ones(6)
            row = [ATOMS.index(label[0]), left, ATOMS.index(label[1]), right, ratio, sign, *gains]
            append(label, f"{label}:{family}:{ratio}", family, row)
    return dict(names=np.array(names), groups=np.array(groups), family=np.array(families),
                context=np.full(len(names), context, dtype=np.int32), recipe=np.array(rows))


def parents(raw, rows):
    left = np.stack([raw[ATOMS[int(row[0])]][int(row[1])] for row in rows])
    right = np.stack([raw[ATOMS[int(row[2])]][int(row[3])] if row[2] >= 0 else left[i]
                      for i, row in enumerate(rows)])
    return left, right


def mix(parents_left, parents_right, names, families, recipes_batch):
    result = parents_left.copy()
    for i, (left, right, label, family, row) in enumerate(zip(
            parents_left, parents_right, names, families, recipes_batch)):
        if family == "pure":
            continue
        ratio, sign = row[4:6]
        if label == "AB" and family in ("propagate", "stress"):
            a = left.astype(np.float64) - left.mean(0, dtype=np.float64)
            b = right.astype(np.float64) - right.mean(0, dtype=np.float64)
            vibration = np.r_[0:6, 9:18]
            rms_a = max(np.sqrt(np.mean(a[:, vibration] ** 2)), 1e-8)
            rms_b = max(np.sqrt(np.mean(b[:, vibration] ** 2)), 1e-8)
            result[i] = a + sign * ratio * rms_a / rms_b * b * np.repeat(row[6:], 3)
        elif label == "AB":
            result[i, :, 9:15] = right[:, 9:15]
        else:
            a, b = left[:, 15:].astype(np.float64), right[:, 15:].astype(np.float64)
            a -= a.mean(0)
            b -= b.mean(0)
            a /= max(np.sqrt(np.mean(a*a)), 1e-8)
            b /= max(np.sqrt(np.mean(b*b)), 1e-8)
            result[i, :, 15:] = (a + sign * ratio * row[-1] * b) / np.sqrt(1 + ratio**2 * row[-1]**2)
    return result


def response_profiles(clean_train):
    profiles = np.zeros((5, 9, 6, 8))
    for index, atom in enumerate(ATOMS):
        for context in range(9):
            selected = (clean_train["names"] == atom) & (clean_train["context"] == context)
            values = clean_train["x"][selected, :1800].reshape(-1, 6, 300)[:, :, :192]
            power = np.expm1(values.reshape(-1, 6, 3, 64).astype(float)).mean((0, 2))
            log_rms = 0.5 * np.log(np.maximum(power.reshape(6, 8, 8).mean(2), 1e-8))
            profiles[index, context] = log_rms - log_rms.mean(1, keepdims=True)
    return profiles


def filtered(signal, knots):
    position = torch.linspace(0, 7, 1025, device=signal.device, dtype=torch.float64)
    left = position.floor().long().clamp_max(6)
    fraction = position - left
    gain = (knots[..., left]*(1-fraction) + knots[..., left+1]*fraction).exp()
    return torch.fft.irfft(torch.fft.rfft(signal, dim=-1)*gain, n=2048, dim=-1)


def perturb_parent(waves, curves, device):
    x = torch.as_tensor(waves, device=device, dtype=torch.float64).transpose(1, 2).reshape(-1, 6, 3, 2048)
    knots = torch.as_tensor(curves, device=device, dtype=torch.float64).unsqueeze(2)
    return filtered(x, knots).reshape(-1, 18, 2048).transpose(1, 2).float().cpu().numpy()


def quadratic(x, strength, time, rms_axes):
    mean = x.mean(time, keepdim=True)
    centered = x - mean
    scale = centered.square().mean(rms_axes, keepdim=True).sqrt().clamp_min(1e-8)
    unit = centered / scale
    y = unit + strength * (unit.square() - unit.square().mean(time, keepdim=True))
    y -= y.mean(time, keepdim=True)
    return y / y.square().mean(rms_axes, keepdim=True).sqrt().clamp_min(1e-8) * scale + mean


def classifier_views(raw, data, profiles, context, part, device, extractor):
    indices = np.arange(0, len(data["names"]), 2)
    rng = np.random.default_rng(1346344912 + context*101 + (11 if part == "train" else 23))
    donors = (context + rng.integers(1, 9, size=(len(indices), 2))) % 9
    levels = rng.uniform(0.5, 1.0, size=(len(indices), 2))
    strengths = rng.uniform(0.03, 0.10, len(indices))
    features = []
    for start in range(0, len(indices), 64):
        chunk = slice(start, start+64)
        ids = indices[chunk]
        rows = data["recipe"][ids]
        left, right = parents(raw, rows)
        transformed = []
        for side, waves in enumerate((left, right)):
            atoms = rows[:, side*2].astype(int)
            atoms = np.where(atoms < 0, rows[:, 0].astype(int), atoms)
            difference = profiles[atoms, donors[chunk, side]] - profiles[atoms, context]
            curves = np.clip(0.5*difference, -np.log(1.5), np.log(1.5)) * levels[chunk, side, None, None]
            transformed.append(perturb_parent(waves, curves, device))
        signal = mix(*transformed, data["names"][ids], data["family"][ids], rows)
        x = torch.as_tensor(signal, device=device, dtype=torch.float64).reshape(-1, 2048, 6, 3)
        eta = torch.as_tensor(strengths[chunk], device=device)[:, None, None, None]
        augmented = quadratic(x, eta, 1, (1, 3)).reshape(-1, 2048, 18).float().cpu().numpy()
        features.append(extractor(augmented, device))
    result = {key: values[indices] for key, values in data.items() if key != "x"}
    result["augmentation"] = np.full(len(indices), "both")
    result["x"] = np.concatenate(features)
    return result


def verifier_view(waves, knots, device):
    x = torch.as_tensor(waves[:, :, 15:], device=device, dtype=torch.float64).transpose(1, 2)
    y = filtered(x, torch.as_tensor(knots, device=device, dtype=torch.float64))
    result = waves.copy()
    result[:, :, 15:] = quadratic(y, 0.06, -1, (1, 2)).transpose(1, 2).float().cpu().numpy()
    return result
