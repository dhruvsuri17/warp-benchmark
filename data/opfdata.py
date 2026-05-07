"""
OPFDataset loader for WARP.

Wraps PyG's OPFDataset and builds heterogeneous typed graphs
with the canonical variable ordering: [Vm, Va, Pg, Qg].
"""

from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from data.transforms import WARPTransform
from data.splits import get_split_indices

CASE_NAME_MAP = {
    "case14": "pglib_opf_case14_ieee",
    "case57": "pglib_opf_case57_ieee",
    "case118": "pglib_opf_case118_ieee",
    "case500": "pglib_opf_case500_goc",
    "case2000": "pglib_opf_case2000_goc",
}


def get_case_name(short_name: str) -> str:
    return CASE_NAME_MAP.get(short_name, short_name)


class OPFDataModule:
    """
    Handles loading OPFData, applying transforms, and providing DataLoaders.

    Usage:
        dm = OPFDataModule(case="case118", split="fulltop", data_root="data/opfdata")
        dm.prepare()
        train_loader = dm.train_loader(batch_size=64)
    """

    def __init__(
        self,
        case: str,
        split: str = "fulltop",
        data_root: str = "data/opfdata",
        seed: int = 42,
        num_workers: int = 4,
    ):
        self.case = case
        self.case_name = get_case_name(case)
        self.split = split
        self.data_root = Path(data_root)
        self.seed = seed
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.transform = None
        self._norm_stats = None

    def prepare(self):
        """Load datasets and compute normalisation statistics from training set."""
        from torch_geometric.datasets import OPFDataset

        topological = self.split == "n-1"

        raw_train = OPFDataset(
            root=str(self.data_root),
            case_name=self.case_name,
            split="train",
            topological_perturbations=topological,
        )
        raw_val = OPFDataset(
            root=str(self.data_root),
            case_name=self.case_name,
            split="val",
            topological_perturbations=topological,
        )
        raw_test = OPFDataset(
            root=str(self.data_root),
            case_name=self.case_name,
            split="test",
            topological_perturbations=topological,
        )

        self.transform = WARPTransform(case_name=self.case_name, data_root=self.data_root)
        self.transform.fit(raw_train)

        self.train_dataset = [self.transform(d) for d in raw_train]
        self.val_dataset = [self.transform(d) for d in raw_val]
        self.test_dataset = [self.transform(d) for d in raw_test]

    def train_loader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_loader(self, batch_size: int = 64) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_loader(self, batch_size: int = 64) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
