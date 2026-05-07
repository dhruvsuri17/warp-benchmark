"""Experiment C: Heavy physics regularization (lam_phy=1.0), slower LR."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(__file__))

import torch
torch.cuda.set_per_process_memory_fraction(0.3)

from colab_warp import train_det, train_warp, benchmark, log, DEVICE

log.info("=" * 60)
log.info("  EXP-C: Heavy physics reg lam_phy=1.0, lr=5e-5, 30+30 epochs")
log.info("=" * 60)

CASE = "pglib_opf_case118_ieee"

det_model = train_det(
    case=CASE, epochs=30, hidden_dim=128, num_layers=6,
    num_groups=5, lr=3e-4,
)
torch.save(det_model.state_dict(), "ckpt/exp_c_det.pt")

warp_model, warp_diff = train_warp(
    case=CASE, epochs=30, hidden_dim=128, num_layers=6,
    num_groups=5, lr=5e-5, lam_phy=1.0,
)
torch.save({"model": warp_model.state_dict(), "diff": warp_diff.state_dict()},
           "ckpt/exp_c_warp.pt")

benchmark(det_model, warp_model, warp_diff, case=CASE, num_groups=1, n_test=50)
log.info("EXP-C COMPLETE")
