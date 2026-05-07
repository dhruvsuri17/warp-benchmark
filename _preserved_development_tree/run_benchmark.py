"""Re-run benchmark only, loading from saved checkpoints."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from colab_warp import (
    DetGNN, HetGNN, GaussianDiffusion, benchmark,
    DEVICE, log
)

HIDDEN_DIM = 128
NUM_LAYERS = 6

det_model = DetGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
det_ckpt = torch.load("ckpt/det_best.pt", map_location=DEVICE, weights_only=True)
det_model.load_state_dict(det_ckpt["model"])
det_model.eval()
log.info(f"Loaded DetGNN (val_loss={det_ckpt['val_loss']:.6f})")

warp_model = HetGNN(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(DEVICE)
warp_diff = GaussianDiffusion().to(DEVICE)
warp_ckpt = torch.load("ckpt/warp_best.pt", map_location=DEVICE, weights_only=True)
warp_model.load_state_dict(warp_ckpt["model"])
warp_diff.load_state_dict(warp_ckpt["diff"])
warp_model.eval()
log.info(f"Loaded WARP (val_ddpm={warp_ckpt['val']:.4f})")

benchmark(det_model, warp_model, warp_diff,
          case="pglib_opf_case118_ieee", num_groups=1, n_test=50)
