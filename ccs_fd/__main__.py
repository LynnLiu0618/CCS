"""Predict fault labels from synchronized windows or precomputed features."""
import argparse
import csv
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np

from . import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NumPy .npy array, one window per row")
    parser.add_argument("--features", action="store_true", help="Read 2628 features per row")
    parser.add_argument("--checkpoint", type=Path, help="Select a different model file")
    parser.add_argument("--device", help="cpu or cuda (default: auto)")
    parser.add_argument("--output", type=Path, help="CSV destination (default: stdout)")
    args = parser.parse_args()
    if args.output and args.input.resolve() == args.output.resolve():
        parser.error("Choose a CSV output different from the input array.")
    data = np.load(args.input, mmap_mode="c", allow_pickle=False)
    model = Predictor(checkpoint=args.checkpoint, device=args.device)
    predict = model.predict if args.features else model.predict_windows
    result = predict(data, return_detail=True)
    destination = args.output.open("w", newline="", encoding="utf-8") if args.output else nullcontext(sys.stdout)
    with destination as stream:
        writer = csv.writer(stream)
        writer.writerow(["window", "candidate", "prediction", "squared_distance"])
        for i, (candidate, label, distance) in enumerate(zip(result["label"], result["pred"], result["distance"])):
            writer.writerow([i, candidate, label, float(distance)])


if __name__ == "__main__":
    main()
