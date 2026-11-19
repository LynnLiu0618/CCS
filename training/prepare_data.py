"""Build CCS-FD source or evaluation features from the BJTU-RAO CSV distribution."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from ccs_fd.features import extract_features
from . import prepare_source as source


CONTEXTS = tuple(f"{hz}Hz_{load}kN" for load in ("0", "+10", "-10") for hz in (20, 40, 60))
CODES = {
    "A": "M1_G0_LA0_RA0", "B": "M0_G3_LA0_RA0", "C": "M0_G0_LA1_RA0",
    "D": "M0_G0_LA2_RA0", "E": "M0_G0_LA4_RA0", "AB": "M1_G3_LA0_RA0",
    "CD": "M0_G0_LA1+LA2_RA0", "F": "M0_G0_LA3_RA0",
}
SENSORS = (("motor", range(1, 10)), ("gearbox", range(11, 17)), ("leftaxlebox", range(17, 20)))
WINDOW = 2048
INTERVALS = {"train": (0, 109), "calibration": (352768, 31), "evaluation": (512768, 62)}


def read_windows(root, context, label, part):
    """Read the named channel columns and exactly the requested half-open interval."""
    start, count = INTERVALS[part]
    code = CODES[label]
    directory = Path(root) / code / f"Sample_{context + 1}"
    chunks = []
    for sensor, channels in SENSORS:
        prefix = f"data_{sensor}_{code}_{CONTEXTS[context]}"
        matches = sorted(directory.glob(prefix + "*.csv"))
        if len(matches) != 1:
            raise ValueError(f"{directory}: expected one CSV matching {prefix}, found {len(matches)}")
        path = matches[0]
        names = [f"CH{i}" for i in channels]
        # Keep the header row; pandas skips the earlier data rows inside its C parser.
        table = pd.read_csv(path, usecols=names, skiprows=range(1, start + 1),
                            nrows=count * WINDOW, dtype=np.float32)
        values = table[names].to_numpy()
        if len(values) != count * WINDOW or not np.isfinite(values).all():
            raise ValueError(f"{path}: expected {count*WINDOW} finite rows starting at sample {start}")
        chunks.append(values)
    return np.concatenate(chunks, axis=1).reshape(count, WINDOW, 18)


def combine(items):
    return {key: np.concatenate([item[key] for item in items]) for key in items[0]}


def write_evaluation(root, output, device):
    pieces = []
    start, count = INTERVALS["evaluation"]
    for context, name in enumerate(CONTEXTS):
        for label in CODES:
            waves = read_windows(root, context, label, "evaluation")
            pieces.append(dict(x=extract_features(waves, device), truth=np.full(count, label),
                               context=np.full(count, name), start=start + WINDOW*np.arange(count, dtype=np.int32)))
        print(f"Prepared evaluation context {name}", flush=True)
    np.savez(output / "evaluation.npz", **combine(pieces))


def clean_features(raw, data, device, knots=None):
    paired = []
    features = []
    for start in range(0, len(data["names"]), 64):
        chunk = slice(start, start+64)
        rows = data["recipe"][chunk]
        left, right = source.parents(raw, rows)
        waves = source.mix(left, right, data["names"][chunk], data["family"][chunk], rows)
        features.append(extract_features(waves, device))
        if knots is not None:
            paired.append(extract_features(source.verifier_view(waves, knots[chunk], device), device))
    clean = dict(data, x=np.concatenate(features), augmentation=np.full(len(data["names"]), "clean"))
    return clean, np.concatenate(paired) if paired else None


def write_source(root, output, device):
    profiles = None
    verification_draws = np.random.default_rng(1346345028)
    for part in ("train", "calibration"):
        clean_rows, classifier_rows, verifier_rows = [], [], []
        for context, name in enumerate(CONTEXTS):
            raw = {label: read_windows(root, context, label, part) for label in source.ATOMS}
            data = source.recipes(part, context)
            knots = None
            if part == "calibration":
                knots = verification_draws.uniform(-np.log(1.5), np.log(1.5), (len(data["names"]), 3, 8))
            clean, paired = clean_features(raw, data, device, knots)
            clean_rows.append(clean)
            if paired is not None:
                verifier_rows.append(paired)
            print(f"Prepared {part} context {name}", flush=True)
        if part == "train":
            profiles = source.response_profiles(combine(clean_rows))
        for context, clean in enumerate(clean_rows):
            raw = {label: read_windows(root, context, label, part) for label in source.ATOMS}
            classifier_rows.append(source.classifier_views(
                raw, clean, profiles, context, part, device, extract_features))
        full = combine(clean_rows + classifier_rows)
        np.savez(output / f"{part}.npz", **full)
        if verifier_rows:
            np.save(output / "calibration_augmented.npy", np.concatenate(verifier_rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "evaluation"))
    parser.add_argument("--data-root", required=True, type=Path, help="Directory containing M*_G*_LA*_RA* folders.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires a CUDA-enabled PyTorch installation and an available GPU")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as temp:
        staging = Path(temp)
        if args.stage == "source":
            write_source(args.data_root, staging, args.device)
        else:
            write_evaluation(args.data_root, staging, args.device)
        for result in sorted(staging.iterdir()):
            result.replace(args.output_dir / result.name)


if __name__ == "__main__":
    main()
