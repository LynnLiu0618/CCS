"""Source-only calibration of five scalar bearing-distance limits."""

import numpy as np

HYPOTHESES = ('H', 'C', 'D', 'E', 'CD')


def fit_limits(clean_distances, augmented_distances, labels):
    """Distances have shape (N, 5); paired rows share their source truth labels."""
    labels = np.asarray(labels)
    hypothesis = np.array([''.join(atom for atom in label if atom in 'CDE') or 'H' for label in labels])
    limits = []
    for column, name in enumerate(HYPOTHESES):
        members = hypothesis == name
        if not members.any():
            raise ValueError(f'Source calibration lacks hypothesis {name}.')
        quantiles = [np.quantile(distances[labels == label, column], .975, method='higher')
                     for distances in (clean_distances, augmented_distances)
                     for label in np.unique(labels[members])]
        limits.append(max(max(quantiles), 1e-8))
    return np.array(limits)
