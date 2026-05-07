"""
Train/val/test split logic for OPFData.

OPFDataset already provides canonical splits, but this module handles
custom splits for ablation studies and reproducibility.
"""

import torch
import numpy as np


def get_split_indices(n_total: int, train_frac: float = 0.8, val_frac: float = 0.1,
                      seed: int = 42):
    """
    Returns (train_idx, val_idx, test_idx) as numpy arrays.
    test_frac = 1 - train_frac - val_frac.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)

    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def stratified_split_by_load(dataset, train_frac=0.8, val_frac=0.1, seed=42, n_bins=10):
    """
    Stratified split based on total load level (sum of Pd).
    Ensures each split sees similar load distribution.
    """
    loads = []
    for data in dataset:
        if hasattr(data, "x_raw"):
            loads.append(data.x_raw[:, 0].sum().item())
        else:
            loads.append(data.x[:, 0].sum().item())

    loads = np.array(loads)
    bins = np.digitize(loads, np.percentile(loads, np.linspace(0, 100, n_bins + 1)[1:-1]))

    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    for b in range(n_bins):
        bin_idx = np.where(bins == b)[0]
        rng.shuffle(bin_idx)
        n = len(bin_idx)
        nt = int(n * train_frac)
        nv = int(n * val_frac)
        train_idx.extend(bin_idx[:nt])
        val_idx.extend(bin_idx[nt:nt + nv])
        test_idx.extend(bin_idx[nt + nv:])

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)
