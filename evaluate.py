"""Evaluate the five frozen CCS-FD models on labeled features or raw windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ccs_fd import Predictor
from ccs_fd.features import extract_features


SEEDS = (1994217509, 1454668731, 1061685734, 577998202, 1919707711)
CHECKPOINTS = Path(__file__).resolve().parent / "ccs_fd" / "checkpoints"
KNOWN = ("A", "B", "C", "D", "E", "AB", "CD")
STATE = {"A": "H", "B": "H", "AB": "H", "C": "C", "D": "D", "E": "E", "CD": "CD"}


def load_inputs(path, labels_path):
    if path.suffix.lower() == ".npy":
        if labels_path is None:
            raise ValueError("Supply a matching --labels labels.npy for an NPY input.")
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        truth = np.load(labels_path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        if labels_path is not None:
            raise ValueError("For NPZ input, include the labels in the same archive.")
        with np.load(path, allow_pickle=False) as archive:
            arrays = [k for k in ("features", "x", "windows") if k in archive]
            labels = [k for k in ("truth", "labels") if k in archive]
            if len(arrays) != 1 or len(labels) != 1:
                raise ValueError("NPZ input needs one of features/x/windows and one of truth/labels.")
            data, truth = archive[arrays[0]], archive[labels[0]]
    else:
        raise ValueError("Expected .npy or .npz input.")

    if data.shape[1:] not in ((2628,), (2048, 18)) or data.dtype.kind not in "iuf":
        raise ValueError("Expected real features (N, 2628) or synchronized windows (N, 2048, 18).")
    if not len(data) or truth.shape != (len(data),) or truth.dtype.kind not in "US":
        raise ValueError("Expected one string label per window and at least one window.")
    truth = truth.astype(str)
    unexpected = set(np.unique(truth)) - set(KNOWN) - {"F"}
    if unexpected:
        raise ValueError(f"Unexpected ground-truth labels: {sorted(unexpected)}. Use F for the benchmark unknown.")
    return data, truth


def rank_auc(positive, scores):
    """Probability that a positive outranks a negative, counting ties as half-wins."""
    pos = int(positive.sum())
    neg = len(positive) - pos
    if not (pos and neg):
        return None
    _, groups, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2
    rank_sum = average_ranks[groups[positive]].sum()
    return float(100 * (rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def window_metrics(truth, predictions, scores):
    unknown = truth == "F"
    correct = predictions == np.where(unknown, "UNKNOWN", truth)

    def percent(values):
        return float(100 * values.mean()) if values.size else None

    return {
        "overall": percent(correct),
        "single": percent(correct[np.isin(truth, KNOWN[:5])]),
        "AB": percent(correct[truth == "AB"]),
        "CD": percent(correct[truth == "CD"]),
        "F": percent(correct[unknown]),
        "known_reject": percent(predictions[~unknown] == "UNKNOWN"),
        "auroc": rank_auc(unknown, scores),
    }


def evaluate_model(predictor, features, truth):
    detail = predictor.predict(features, return_detail=True)
    cutoffs = dict(zip(predictor.pack["hypotheses"], predictor.pack["thresholds"]))
    limits = np.asarray([float(cutoffs[STATE[candidate]]) for candidate in detail["label"]])
    scores = detail["distance"] / limits
    if not np.isfinite(scores).all():
        raise ValueError("The model produced a non-finite normalized distance.")
    return window_metrics(truth, detail["pred"], scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ with features/x/windows and truth/labels, or NPY data")
    parser.add_argument("--labels", type=Path, help="NPY string labels for NPY input")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Each seed must be listed only once.")
    if args.output and args.output.resolve() in {p.resolve() for p in (args.input, args.labels) if p}:
        parser.error("The report path must differ from the input data and labels.")
    try:
        data, truth = load_inputs(args.input, args.labels)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    features = extract_features(data, args.device) if data.ndim == 3 else data
    runs = {}
    for seed in args.seeds:
        predictor = Predictor(CHECKPOINTS / f"{seed}.pt", device=args.device)
        runs[seed] = evaluate_model(predictor, features, truth)

    summary = {}
    for metric in runs[args.seeds[0]]:
        values = [run[metric] for run in runs.values()]
        defined = values[0] is not None
        summary[metric] = {
            "mean": float(np.mean(values)) if defined else None,
            "sample_sd": float(np.std(values, ddof=1)) if defined and len(values) > 1 else None,
        }
    labels, counts = np.unique(truth, return_counts=True)
    report = {
        "units": "percent",
        "windows": len(truth),
        "label_counts": dict(zip(labels.tolist(), counts.tolist())),
        "runs": runs,
        "summary": summary,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
