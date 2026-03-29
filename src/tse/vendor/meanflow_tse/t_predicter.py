"""TPredicter: predicts the mixing ratio (alpha) for MeanFlow-TSE.

Vendored from: https://github.com/rikishimizu/MeanFlow-TSE/blob/main/models/t_predicter.py
License: MIT

Takes mixture + enrollment waveforms, extracts ECAPA-TDNN embeddings for both,
concatenates, and predicts a scalar alpha via sigmoid.
"""

import torch
import torch.nn as nn

from .ecapa_tdnn import ECAPA_TDNN


class TPredicter(nn.Module):
    def __init__(self, C):
        super(TPredicter, self).__init__()
        self.ecapa_tdnn = ECAPA_TDNN(C=C)
        self.output_activ = nn.Sigmoid()
        self.output_layer = nn.Sequential(
            nn.Linear(192 * 2, 192),
            nn.SiLU(),
            nn.Linear(192, 1),
        )

    def forward(self, mixture, enrollment, aug=False):
        """
        Args:
            mixture: (batch_size, time_steps) waveform
            enrollment: (batch_size, time_steps) waveform

        Returns:
            t: (batch_size,) predicted mixing ratio
        """
        enrollment_feat = self.ecapa_tdnn(enrollment, aug)
        mixture_feat = self.ecapa_tdnn(mixture, aug)
        sqrt_d = enrollment_feat.shape[1] ** 0.5
        enrollment_feat = enrollment_feat / sqrt_d
        mixture_feat = mixture_feat / sqrt_d

        similarity = torch.cat([enrollment_feat, mixture_feat], dim=-1)
        similarity = self.output_layer(similarity).squeeze(-1)
        t = self.output_activ(similarity)

        return t
