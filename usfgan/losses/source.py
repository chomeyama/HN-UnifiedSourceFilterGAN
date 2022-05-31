# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Source regularization loss modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import usfgan.losses
from librosa.filters import mel as librosa_mel
from usfgan.layers import CheapTrick


class FlattenLoss(nn.Module):
    # Spectral envelope flattening loss module.

    def __init__(
        self,
        sampling_rate=24000,
        hop_size=120,
        fft_size=2048,
        f0_floor=70,
        f0_ceil=340,
        power=False,
        elim_0th=False,
        l2_norm=False,
    ):
        """Initialize spectral envelope regularlization loss module.

        Args:
            sampling_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum F0 value.
            f0_ceil (int): Maximum F0 value.
            uv_threshold (float): V/UV determining threshold.
            q1 (float): Parameter to remove effect of adjacent harmonics.

        """
        super(FlattenLoss, self).__init__()
        self.hop_size = hop_size
        self.power = power
        self.elim_0th = elim_0th
        self.cheaptrick = CheapTrick(
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            fft_size=fft_size,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        if l2_norm:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

    def forward(self, s, f):
        """Calculate forward propagation.

        Args:
            s (Tensor): Predicted source signal (B, 1, T).
            f (Tensor): Extracted F0 sequence (B, 1, T').

        Returns:
            loss (Tensor): Spectral envelope regularization loss value.

        """
        s, f = s.squeeze(1), f.squeeze(1)
        e = self.cheaptrick.forward(s, f, self.power, self.elim_0th)
        loss = self.loss(e, e.new_zeros(e.size()))
        return loss


class ResidualLoss(nn.Module):
    # hn-uSFGAN source regularization loss module.

    def __init__(
        self,
        sampling_rate=24000,
        fft_size=2048,
        hop_size=120,
        f0_floor=70,
        f0_ceil=340,
        n_mels=80,
        fmin=0,
        fmax=None,
        power=False,
        elim_0th=True,
    ):
        """Initialize Mel-spectrogram loss.

        Args:
            sampling_rate (int): Sampling rate.
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            f0_floor (int): Minimum F0 value.
            f0_ceil (int): Maximum F0 value.
            n_mels (int): Number of Mel basis.
            fmin (int): Minimum frequency for Mel.
            fmax (int): Maximum frequency for Mel.
            power (bool): Whether to use power or magnitude spectrogram.
            elim_0th (bool): Whether to exclude 0th cepstrum in CheapTrick.
                If set to true, power is estimated by source-network.

        """
        super(ResidualLoss, self).__init__()
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.cheaptrick = CheapTrick(
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            fft_size=fft_size,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        self.win_length = fft_size
        self.register_buffer("window", torch.hann_window(self.win_length))

        # define mel-filter-bank
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sampling_rate / 2
        melmat = librosa_mel(
            sr=sampling_rate, n_fft=fft_size, n_mels=n_mels, fmin=fmin, fmax=self.fmax
        ).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float())

        self.power = power
        self.elim_0th = elim_0th

    def forward(self, s, y, f):
        """Calculate forward propagation.

        Args:
            s (Tensor): Predicted source signal (B, 1, T).
            y (Tensor): Target signal (B, 1, T).
            f (Tensor): Extracted F0 sequence (B, 1, T').

        Returns:
            Tensor: Difference loss value.

        """
        s, y, f = s.squeeze(1), y.squeeze(1), f.squeeze(1)

        with torch.no_grad():
            # calculate log power (or magnitude) spectrograms
            e = self.cheaptrick.forward(y, f, self.power, self.elim_0th)
            y = usfgan.losses.stft(
                y,
                self.fft_size,
                self.hop_size,
                self.win_length,
                self.window,
                power=self.power,
            )
            # adjust length, (B, T', C)
            minlen = min(e.size(1), y.size(1))
            e, y = e[:, :minlen, :], y[:, :minlen, :]

            # calculate mean power (or magnitude) of y
            if self.elim_0th:
                y_mean = y.mean(dim=-1, keepdim=True)

            # calculate target of output source signal
            y = torch.log(torch.clamp(y, min=1e-7))
            t = (y - e).exp()
            if self.elim_0th:
                t_mean = t.mean(dim=-1, keepdim=True)
                t = y_mean / t_mean * t

            # apply mel-filter-bank and log
            t = torch.matmul(t, self.melmat)
            t = torch.log(torch.clamp(t, min=1e-7))

        # calculate power (or magnitude) spectrogram
        s = usfgan.losses.stft(
            s,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            power=self.power,
        )
        # adjust length, (B, T', C)
        minlen = min(minlen, s.size(1))
        s, t = s[:, :minlen, :], t[:, :minlen, :]

        # apply mel-filter-bank and log
        s = torch.matmul(s, self.melmat)
        s = torch.log(torch.clamp(s, min=1e-7))

        loss = F.l1_loss(s, t.detach())

        return loss
