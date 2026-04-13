"""Vendored MeanFlow-TSE inference code.

Source: https://github.com/rikishimizu/MeanFlow-TSE
Paper: https://arxiv.org/abs/2512.18572
License: MIT

Only inference-critical files are vendored. Training-only dependencies
(pytorch-lightning, asteroid, hydra, wandb) are eliminated.
"""

from .udit_meanflow.udit_meanflow import UDiT
from .ecapa_tdnn import ECAPA_TDNN
from .t_predicter import TPredicter
from .transforms import stft_torch, istft_torch
