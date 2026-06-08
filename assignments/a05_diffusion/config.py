"""Hyperparameters for the tiny diffusion model.

Everything is deliberately small: 1-channel 16x16 images, a base-width-32 U-Net, three
shape classes. The point of A5 is the diffusion mechanism (schedules, the q_sample closed
form, the eps/x0/v parameterizations, DDPM/DDIM sampling, classifier-free guidance), not
a competitive image model, so the network is a small stand-in that overfits one batch and
trains to visibly-improving samples on a 4080.

T is the schedule length for a real training run. The tests pass a much smaller T (the
cosine schedule is self-normalizing in T; the linear schedule's beta constants are
calibrated for T=1000 and do not transfer, so tests only assert the endpoint on cosine).
"""

from dataclasses import dataclass


@dataclass
class DiffusionConfig:
    img_size: int = 16
    channels: int = 1
    num_classes: int = 3       # disk, square, cross; the U-Net adds one extra null row

    T: int = 1000              # schedule length for real training
    base_width: int = 32       # U-Net base channel width; time MLP width is 4x this
    cfg_drop_prob: float = 0.1  # probability of dropping the label to null during training
    min_snr_gamma: float = 5.0  # Min-SNR loss-weighting cap (Hang et al. 2023)
