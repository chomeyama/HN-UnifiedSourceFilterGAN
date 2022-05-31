# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules.

References:
    - https://github.com/bigpon/QPPWG
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

from logging import getLogger
from multiprocessing import Manager

import numpy as np
import soundfile as sf
from hydra.utils import to_absolute_path
from joblib import load
from torch.utils.data import Dataset
from usfgan.utils import (
    check_filename,
    dilated_factor,
    read_hdf5,
    read_txt,
    validate_length,
)

# A logger for this file
logger = getLogger(__name__)


class AudioFeatDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(
        self,
        stats,
        audio_list,
        feat_list,
        audio_length_threshold=None,
        feat_length_threshold=None,
        return_filename=False,
        allow_cache=False,
        sample_rate=24000,
        hop_size=120,
        dense_factor=4,
        df_f0_type="contf0",
        aux_feats=["mcep", "mcap"],
    ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            audio_list (str): Filename of the list of audio files.
            feat_list (str): Filename of the list of feature files.
            audio_length_threshold (int): Threshold to remove short audio files.
            feat_length_threshold (int): Threshold to remove short feature files.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            hop_size (int): Hope size of acoustic feature
            dense_factor (int): Number of taps in one cycle.
            aux_feats (str): Type of auxiliary features.

        """
        # load audio and feature files & check filename
        audio_files = read_txt(to_absolute_path(audio_list))
        feat_files = read_txt(to_absolute_path(feat_list))
        assert check_filename(audio_files, feat_files)

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [sf.read(to_absolute_path(f)).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]
        if feat_length_threshold is not None:
            f0_lengths = [
                read_hdf5(to_absolute_path(f), "/f0").shape[0] for f in feat_files
            ]
            idxs = [
                idx
                for idx in range(len(feat_files))
                if f0_lengths[idx] > feat_length_threshold
            ]
            if len(feat_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(feat_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"${audio_list} is empty."
        assert len(audio_files) == len(
            feat_files
        ), f"Number of audio and features files are different ({len(audio_files)} vs {len(feat_files)})."

        self.audio_files = audio_files
        self.feat_files = feat_files
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.dense_factor = dense_factor
        self.aux_feats = aux_feats
        self.df_f0_type = df_f0_type
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

        # define feature pre-processing function
        self.scaler = load(stats)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Audio signal (T,).
            ndarray: UV feature (T', C).
            ndarray: F0 feature (T', C).
            ndarray: Mel-spectrogram feature (T', C).
            ndarray: Auxiliary feature (T', C).
            ndarray: Dilated factor (T, 1).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        # load audio and features
        audio, sr = sf.read(to_absolute_path(self.audio_files[idx]))
        # audio & feature pre-processing
        audio = audio.astype(np.float32)

        # get auxiliary features
        aux_feats = []
        for feat_type in self.aux_feats:
            aux_feat = read_hdf5(
                to_absolute_path(self.feat_files[idx]), f"/{feat_type}"
            )
            if feat_type != "uv":
                aux_feat = self.scaler[f"{feat_type}"].transform(aux_feat)
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1)

        # adjust length
        audio, aux_feats = validate_length(audio, aux_feats, self.hop_size)

        # get dilated factor sequence
        f0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/f0")  # descrete F0
        contf0 = read_hdf5(
            to_absolute_path(self.feat_files[idx]), "/contf0"
        )  # continuous F0
        aux_feats, f0 = validate_length(aux_feats, f0)
        f0, contf0 = validate_length(f0, contf0)
        if self.df_f0_type == "contf0":
            df = dilated_factor(
                np.squeeze(contf0.copy()), self.sample_rate, self.dense_factor
            )
        else:
            df = dilated_factor(
                np.squeeze(f0.copy()), self.sample_rate, self.dense_factor
            )
        df = df.repeat(self.hop_size, axis=0)

        if self.return_filename:
            items = self.feat_files[idx], audio, aux_feats, df, f0, contf0
        else:
            items = audio, aux_feats, df, f0, contf0

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class FeatDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        stats,
        feat_list,
        feat_length_threshold=None,
        return_filename=False,
        allow_cache=False,
        sample_rate=24000,
        hop_size=120,
        dense_factor=4,
        df_f0_type="contf0",
        aux_feats=["mcep", "mcap"],
        f0_factor=1.0,
    ):
        """Initialize dataset.

        Args:
            stats (str): Filename of the statistic hdf5 file.
            feat_list (str): Filename of the list of feature files.
            feat_length_threshold (int): Threshold to remove short feature files.
            return_filename (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
            hop_size (int): Hope size of acoustic feature
            dense_factor (int): Number of taps in one cycle.
            aux_feats (str): Type of auxiliary features.
            f0_factor (float): Ratio of scaled f0.

        """
        # load feat. files
        feat_files = read_txt(to_absolute_path(feat_list))

        # filter by threshold
        if feat_length_threshold is not None:
            f0_lengths = [
                read_hdf5(to_absolute_path(f), "/f0").shape[0] for f in feat_files
            ]
            idxs = [
                idx
                for idx in range(len(feat_files))
                if f0_lengths[idx] > feat_length_threshold
            ]
            if len(feat_files) != len(idxs):
                logger.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(feat_files)} -> {len(idxs)})."
                )
            feat_files = [feat_files[idx] for idx in idxs]

        # assert the number of files
        assert len(feat_files) != 0, f"${feat_list} is empty."

        self.feat_files = feat_files
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.dense_factor = dense_factor
        self.df_f0_type = df_f0_type
        self.aux_feats = aux_feats
        self.f0_factor = f0_factor
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(feat_files))]

        # define feature pre-processing function
        self.scaler = load(stats)

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: UV feature (T', C).
            ndarray: F0 feature (T', C).
            ndarray: Mel-spectrogram feature (T', C).
            ndarray: Auxiliary feature (T', C).
            ndarray: Dilated factor (T, 1).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        # get auxiliary features
        aux_feats = []
        for feat_type in self.aux_feats:
            aux_feat = read_hdf5(
                to_absolute_path(self.feat_files[idx]), f"/{feat_type}"
            )
            if "f0" in feat_type:
                aux_feat *= self.f0_factor
            if feat_type != "uv":
                aux_feat = self.scaler[f"{feat_type}"].transform(aux_feat)
            aux_feats += [aux_feat]
        aux_feats = np.concatenate(aux_feats, axis=1)

        # get dilated factor sequence
        f0 = read_hdf5(to_absolute_path(self.feat_files[idx]), "/f0")  # descrete F0
        contf0 = read_hdf5(
            to_absolute_path(self.feat_files[idx]), "/contf0"
        )  # continuous F0
        f0 *= self.f0_factor
        contf0 *= self.f0_factor
        aux_feats, f0 = validate_length(aux_feats, f0)
        f0, contf0 = validate_length(f0, contf0)
        if self.df_f0_type == "contf0":
            df = dilated_factor(
                np.squeeze(contf0.copy()), self.sample_rate, self.dense_factor
            )
        else:
            df = dilated_factor(
                np.squeeze(f0.copy()), self.sample_rate, self.dense_factor
            )
        df = df.repeat(self.hop_size, axis=0)

        if self.return_filename:
            items = self.feat_files[idx], aux_feats, df, f0, contf0
        else:
            items = aux_feats, df, f0, contf0

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.feat_files)
