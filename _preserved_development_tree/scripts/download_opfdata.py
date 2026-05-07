"""
Download OPFData dataset via PyG.

Usage:
    python scripts/download_opfdata.py --case case118
    python scripts/download_opfdata.py --case case118 --n1
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


CASE_NAMES = {
    "case14": "pglib_opf_case14_ieee",
    "case57": "pglib_opf_case57_ieee",
    "case118": "pglib_opf_case118_ieee",
    "case500": "pglib_opf_case500_goc",
    "case2000": "pglib_opf_case2000_goc",
}


def download(case: str, n1: bool = False, root: str = "data/opfdata"):
    from torch_geometric.datasets import OPFDataset

    case_name = CASE_NAMES.get(case, case)
    logger.info(f"Downloading {case_name} (N-1={n1}) to {root}...")

    for split in ["train", "val", "test"]:
        logger.info(f"  Loading {split} split...")
        dataset = OPFDataset(
            root=root,
            case_name=case_name,
            split=split,
            topological_perturbations=n1,
        )
        logger.info(f"  {split}: {len(dataset)} instances")
        logger.info(f"  Sample: x.shape={dataset[0].x.shape}, "
                     f"edge_index.shape={dataset[0].edge_index.shape}")

    logger.info("Download complete.")


def main():
    parser = argparse.ArgumentParser(description="Download OPFData")
    parser.add_argument("--case", type=str, default="case118",
                        choices=list(CASE_NAMES.keys()))
    parser.add_argument("--n1", action="store_true", help="Download N-1 split")
    parser.add_argument("--root", type=str, default="data/opfdata")
    args = parser.parse_args()

    download(args.case, args.n1, args.root)


if __name__ == "__main__":
    main()
