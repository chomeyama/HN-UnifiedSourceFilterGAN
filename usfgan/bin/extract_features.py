# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature extraction script.

References:
    - https://github.com/bigpon/QPPWG
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/k2kobayashi/sprocket

"""

import copy
import multiprocessing as mp
import os
import sys
from logging import getLogger

import hydra
import librosa
import numpy as np
import pysptk
import pyworld
import soundfile as sf
import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter
from usfgan.utils import read_txt, write_hdf5

# A logger for this file
logger = getLogger(__name__)


def path_create(wav_list, in_dir, out_dir, extname):
    for wav_name in wav_list:
        path_replace(wav_name, in_dir, out_dir, extname=extname)


def path_replace(filepath, inputpath, outputpath, extname=None):
    if extname is not None:
        filepath = f"{filepath.split('.')[0]}.{extname}"
    filepath = filepath.replace(inputpath, outputpath)
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    return filepath


def spk_division(file_list, config, spkinfo, split="/"):
    """Divide list into speaker-dependent list

    Args:
        file_list (list): Waveform list
        config (dict): Config
        spkinfo (dict): Dictionary of
            speaker-dependent f0 range and power threshold
        split: Path split string

    Return:
        (list): List of divided file lists
        (list): List of speaker-dependent configs

    """
    file_lists, configs, tempf = [], [], []
    prespk = None
    for file in file_list:
        spk = file.split(split)[config.spkidx]
        if spk != prespk:
            if tempf:
                file_lists.append(tempf)
            tempf = []
            prespk = spk
            tempc = copy.deepcopy(config)
            if spk in spkinfo:
                tempc["minf0"] = spkinfo[spk]["f0_min"]
                tempc["maxf0"] = spkinfo[spk]["f0_max"]
                tempc["pow_th"] = spkinfo[spk]["pow_th"]
            else:
                msg = f"Since {spk} is not in spkinfo dict, "
                msg += "default f0 range and power threshold are used."
                logger.info(msg)
                tempc["minf0"] = 70
                tempc["maxf0"] = 300
                tempc["pow_th"] = -20
            configs.append(tempc)
        tempf.append(file)
    file_lists.append(tempf)

    return file_lists, configs


def aux_list_create(wav_list_file, config):
    """Create list of auxiliary acoustic features

    Args:
        wav_list_file (str): Filename of wav list
        config (dict): Config

    """
    aux_list_file = wav_list_file.replace(".scp", ".list")
    wav_files = read_txt(wav_list_file)
    with open(aux_list_file, "w") as f:
        for wav_name in wav_files:
            feat_name = path_replace(
                wav_name,
                config.in_dir,
                config.out_dir,
                extname=config.feature_format,
            )
            f.write(f"{feat_name}\n")


def low_cut_filter(x, fs, cutoff=70):
    """Low cut filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def low_pass_filter(x, fs, cutoff=70, padding=True):
    """Low pass filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), "edge")
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2 : -numtaps // 2]

    return lpf_x


def convert_continuos_f0(f0):
    """Convert F0 to continuous F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)

    """
    # get uv information as binary
    uv = np.float32(f0 != 0)
    # get start and end of f0
    if (f0 == 0).all():
        logger.warn("all of the f0 values are 0.")
        return uv, f0, False
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cont_f0 = copy.deepcopy(f0)
    start_idx = np.where(cont_f0 == start_f0)[0][0]
    end_idx = np.where(cont_f0 == end_f0)[0][-1]
    cont_f0[:start_idx] = start_f0
    cont_f0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cont_f0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cont_f0[nz_frames])
    cont_f0 = f(np.arange(0, cont_f0.shape[0]))

    return uv, cont_f0, True


# Mel-spectrogram and F0 features
def melfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
):
    """Extract linear mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.

    Returns:
        ndarray: Linear mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )

    return np.dot(spc, mel_basis.T)


def melf0_feature_extract(queue, wav_list, config, eps=1e-7):
    """Mel-spc w/ discrete F0 and continuous F0 feature extraction

    Args:
        queue (multiprocessing.Queue): the queue to store the file name of utterance
        wav_list (list): list of the wav files
        config (dict): feature extraction config

    """
    # extraction
    for i, wav_name in enumerate(wav_list):
        logger.info(f"now processing {wav_name} ({i + 1}/{len(wav_list)})")

        # load wavfile
        x, fs = sf.read(to_absolute_path(wav_name))
        x = np.array(x, dtype=np.float)

        # check sampling frequency
        if not fs == config.sampling_rate:
            logger.error("sampling frequency is not matched.")
            sys.exit(1)

        # apply low-cut-filter
        if config.highpass_cutoff > 0:
            if (x == 0).all():
                logger.info(f"xxxxx {wav_name}")
                continue
            x = low_cut_filter(x, fs, cutoff=config.highpass_cutoff)

        # extract WORLD features
        f0, t = pyworld.harvest(
            x,
            fs=config.sampling_rate,
            f0_floor=config.minf0,
            f0_ceil=config.maxf0,
            frame_period=config.shiftms,
        )
        env = pyworld.cheaptrick(
            x,
            f0,
            t,
            fs=config.sampling_rate,
            fft_size=config.fft_size,
        )
        ap = pyworld.d4c(
            x,
            f0,
            t,
            fs=config.sampling_rate,
            fft_size=config.fft_size,
        )
        uv, cont_f0, is_all_uv = convert_continuos_f0(f0)
        if is_all_uv:
            lpf_fs = int(config.sampling_rate / config.hop_size)
            cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=20)
            next_cutoff = 70
            while not (cont_f0_lpf >= [0]).all():
                cont_f0_lpf = low_pass_filter(cont_f0, lpf_fs, cutoff=next_cutoff)
                next_cutoff *= 2
        else:
            cont_f0_lpf = cont_f0
            logger.warn(f"all of the f0 values are 0 {wav_name}.")
        mcep = pysptk.sp2mc(env, order=config.mcep_dim, alpha=config.alpha)
        mcap = pysptk.sp2mc(ap, order=config.mcap_dim, alpha=config.alpha)
        codeap = pyworld.code_aperiodicity(ap, config.sampling_rate)

        # extract mel-spectrogram
        msp = melfilterbank(
            x,
            config.sampling_rate,
            fft_size=config.fft_size,
            hop_size=config.hop_size,
            win_length=config.win_length,
            window=config.window,
            num_mels=config.num_mels,
            fmin=config.fmin,
            fmax=config.fmax,
        )

        # adjust shapes
        minlen = min(uv.shape[0], msp.shape[0])
        uv = uv[:minlen]
        uv = np.expand_dims(uv, axis=-1)
        f0 = f0[:minlen]
        f0 = np.expand_dims(f0, axis=-1)
        cont_f0_lpf = cont_f0_lpf[:minlen]
        cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
        mcep = mcep[:minlen]
        mcap = mcap[:minlen]
        codeap = codeap[:minlen]
        logmsp = np.log10(np.maximum(eps, msp))

        # save features
        feat_name = path_replace(
            wav_name,
            config.in_dir,
            config.out_dir,
            extname=config.feature_format,
        )
        write_hdf5(to_absolute_path(feat_name), "/uv", uv)
        write_hdf5(to_absolute_path(feat_name), "/f0", f0)
        write_hdf5(to_absolute_path(feat_name), "/contf0", cont_f0_lpf)
        write_hdf5(to_absolute_path(feat_name), "/mcep", mcep)
        write_hdf5(to_absolute_path(feat_name), "/mcap", mcap)
        write_hdf5(to_absolute_path(feat_name), "/codeap", codeap)
        write_hdf5(to_absolute_path(feat_name), "/logmsp", logmsp)

    queue.put("Finish")


@hydra.main(version_base=None, config_path="config", config_name="extract_features")
def main(config: DictConfig):
    # show argument
    logger.info(OmegaConf.to_yaml(config))

    # read list
    file_list = read_txt(to_absolute_path(config.audio))
    logger.info(f"number of utterances = {len(file_list)}")

    # list division
    if config.spkinfo and os.path.exists(to_absolute_path(config.spkinfo)):
        # load speaker info
        with open(to_absolute_path(config.spkinfo), "r") as f:
            spkinfo = yaml.safe_load(f)
        logger.info(f"Spkinfo {config.spkinfo} is used.")
        # divide into each spk list
        file_lists, configs = spk_division(file_list, config, spkinfo)
    else:
        logger.info(
            f"Since spkinfo {config.spkinfo} is not exist, default f0 range and power threshold are used."
        )
        file_lists = np.array_split(file_list, 10)
        file_lists = [f_list.tolist() for f_list in file_lists]
        configs = [config] * len(file_lists)

    # set mode
    if config.inv:
        target_fn = melf0_feature_extract
        # create auxiliary feature list
        aux_list_create(to_absolute_path(config.audio), config)
        # create folder
        path_create(file_list, config.in_dir, config.out_dir, config.feature_format)

    # multi processing
    processes = []
    queue = mp.Queue()
    for f, _config in zip(file_lists, configs):
        p = mp.Process(
            target=target_fn,
            args=(queue, f, _config),
        )
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
