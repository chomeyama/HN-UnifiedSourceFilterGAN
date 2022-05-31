# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decoding Script for Unified Source-Filter GAN.

References:
    - https://github.com/bigpon/QPPWG
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import os
from logging import getLogger
from time import time

import hydra
import numpy as np
import soundfile as sf
import torch
import usfgan
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from usfgan.datasets import FeatDataset
from usfgan.utils.features import SignalGenerator

# A logger for this file
logger = getLogger(__name__)


@hydra.main(config_path="config", config_name="decode")
def main(config: DictConfig) -> None:
    """Run decoding process."""

    # fix seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load pre-trained model from checkpoint file
    model = {
        "generator": hydra.utils.instantiate(config.generator),
    }
    model["generator"].load_state_dict(
        torch.load(to_absolute_path(config.checkpoint_path))["model"]["generator"]
    )
    logger.info(f"Loaded model parameters from {config.checkpoint_path}.")
    model["generator"].remove_weight_norm()
    model["generator"] = model["generator"].eval().to(device)

    # check directory existence
    out_dir = to_absolute_path(config.out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # get dataset
    dataset = FeatDataset(
        stats=to_absolute_path(config.data.stats),
        feat_list=config.data.eval_feat,
        return_filename=True,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        dense_factor=config.data.dense_factor,
        df_f0_type=config.data.df_f0_type,
        aux_feats=config.data.aux_feats,
        f0_factor=config.f0_factor,
    )
    logger.info(f"The number of features to be decoded = {len(dataset)}.")

    # get data processor
    signal_generator = SignalGenerator(
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        sine_amp=config.data.sine_amp,
        noise_amp=config.data.noise_amp,
        signal_types=config.data.signal_types,
    )
    pad_fn = torch.nn.ReplicationPad1d(config.generator.aux_context_window)

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, (feat_path, c, df, f0, contf0) in enumerate(pbar, 1):
            # setup input features
            c = pad_fn(torch.FloatTensor(c).unsqueeze(0).transpose(2, 1)).to(device)
            df = torch.FloatTensor(df).view(1, 1, -1).to(device)
            f0 = torch.FloatTensor(f0).unsqueeze(0).transpose(2, 1).to(device)
            contf0 = torch.FloatTensor(contf0).unsqueeze(0).transpose(2, 1).to(device)
            # create input signal tensor
            if config.data.sine_f0_type == "contf0":
                in_signal = signal_generator(contf0)
            else:
                in_signal = signal_generator(f0)

            # generate
            start = time()
            y, s = model["generator"](in_signal, c, df)[:2]
            rtf = (time() - start) / (y.size(-1) / config.data.sample_rate)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save output signal as PCM 16 bit wav file
            utt_id = os.path.splitext(os.path.basename(feat_path))[0]
            if config.f0_factor == 1.0:
                wav_filename = f"{utt_id}.wav"
            else:  # scaled f0
                wav_filename = f"{utt_id}_f{config.f0_factor:.2f}.wav"
            print(wav_filename)
            sf.write(
                os.path.join(out_dir, wav_filename),
                y.view(-1).cpu().numpy(),
                config.data.sample_rate,
                "PCM_16",
            )

            # save source signal as PCM 16 bit wav file
            if config.save_source:
                wav_filename = wav_filename.replace(".wav", "_s.wav")
                s = s.view(-1).cpu().numpy()
                s = s / np.max(np.abs(s))  # normalize
                sf.write(
                    os.path.join(out_dir, wav_filename),
                    s,
                    config.data.sample_rate,
                    "PCM_16",
                )

    # report average RTF
    logger.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()
