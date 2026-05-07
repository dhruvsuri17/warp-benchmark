"""
Learning rate and noise schedules for WARP training.

- Cosine LR schedule with linear warmup
- Noise schedule is in models/diffusion.py
"""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int,
                                    num_training_steps: int,
                                    min_lr_ratio: float = 0.0):
    """
    Cosine LR schedule with linear warmup.

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: steps for linear warmup
        num_training_steps: total training steps
        min_lr_ratio: minimum LR as fraction of peak LR
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)
