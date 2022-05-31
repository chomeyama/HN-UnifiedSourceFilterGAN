# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based loss modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel


def stft(
    x, fft_size, hop_size, win_length, window, center=True, onesided=True, power=False
):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        center=center,
        onesided=onesided,
        return_complex=False,
    )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    if power:
        return torch.clamp(real ** 2 + imag ** 2, min=1e-7).transpose(2, 1)
    else:
        return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize log STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, hop_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(x, self.fft_size, self.hop_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.hop_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, 1, T).
            y (Tensor): Groundtruth signal (B, 1, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        x = x.squeeze(1)
        y = y.squeeze(1)

        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class LogSTFTPowerLoss(nn.Module):
    """Log STFT power loss module."""

    def __init__(
        self, fft_size=1024, hop_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(LogSTFTPowerLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_pow = stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            power=True,
        )
        y_pow = stft(
            y,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            power=True,
        )
        stft_loss = (
            self.mse(
                torch.log(torch.clamp(x_pow, min=1e-7)),
                torch.log(torch.clamp(y_pow, min=1e-7)),
            )
            / 2.0
        )

        return stft_loss


class MultiResolutionLogSTFTPowerLoss(nn.Module):
    """Multi-resolution log STFT power loss module.

    This loss is same as the loss of Neural Source-Filter.
    https://arxiv.org/abs/1904.12088
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi-resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionLogSTFTPowerLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [LogSTFTPowerLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Multi-resolution log STFT power loss value.

        Args:
            x (Tensor): Predicted signal (B, 1, T).
            y (Tensor): Groundtruth signal (B, 1, T).

        Returns:
            Tensor: Multi-resolution log STFT power loss value.

        """
        x = x.squeeze(1)
        y = y.squeeze(1)

        stft_loss = 0.0
        for f in self.stft_losses:
            l = f(x, y)
            stft_loss += l
        stft_loss /= len(self.stft_losses)

        return stft_loss


class MelSpectralLoss(nn.Module):
    """Mel-spectral L1 loss module."""

    def __init__(
        self,
        fft_size=1024,
        hop_size=120,
        win_length=1024,
        window="hann_window",
        sampling_rate=24000,
        n_mels=80,
        fmin=0,
        fmax=None,
    ):
        """Initialize MelSpectralLoss loss.

        Args:
            fft_size (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            sampling_rate (int): Sampling rate.
            n_mels (int): Number of Mel basis.
            fmin (Optional[int]): Minimum frequency of mel-filter-bank.
            fmax (Optional[int]): Maximum frequency of mel-filter-bank.

        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length if win_length is not None else fft_size
        self.register_buffer("window", getattr(torch, window)(self.win_length))
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sampling_rate / 2
        melmat = librosa_mel(
            sr=sampling_rate, n_fft=fft_size, n_mels=n_mels, fmin=fmin, fmax=fmax
        ).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float())

    def forward(self, x, y):
        """Calculate Mel-spectral L1 loss.

        Args:
            x (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).

        Returns:
            Tensor: Mel-spectral L1 loss value.

        """
        x = x.squeeze(1)
        y = y.squeeze(1)
        x_mag = stft(x, self.fft_size, self.hop_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.hop_size, self.win_length, self.window)
        x_log_mel = torch.log(torch.clamp(torch.matmul(x_mag, self.melmat), min=1e-7))
        y_log_mel = torch.log(torch.clamp(torch.matmul(y_mag, self.melmat), min=1e-7))
        mel_loss = F.l1_loss(x_log_mel, y_log_mel)

        return mel_loss
