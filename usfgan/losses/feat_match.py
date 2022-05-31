# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature matching loss module.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import torch.nn as nn
import torch.nn.functional as F


class FeatureMatchLoss(nn.Module):
    # Feature matching loss module.

    def __init__(
        self,
        average_by_layers=False,
    ):
        """Initialize FeatureMatchLoss module."""
        super(FeatureMatchLoss, self).__init__()
        self.average_by_layers = average_by_layers

    def forward(self, fmaps_fake, fmaps_real):
        """Calculate forward propagation.

        Args:
            fmaps_fake (list): List of discriminator outputs
                calcuated from generater outputs.
            fmaps_real (list): List of discriminator outputs
                calcuated from groundtruth.

        Returns:
            Tensor: Feature matching loss value.

        """
        feat_match_loss = 0.0
        for feat_fake, feat_real in zip(fmaps_fake, fmaps_real):
            feat_match_loss += F.l1_loss(feat_fake, feat_real.detach())

        if self.average_by_layers:
            feat_match_loss /= len(fmaps_fake)

        return feat_match_loss
