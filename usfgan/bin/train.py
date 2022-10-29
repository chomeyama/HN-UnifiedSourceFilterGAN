# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Training Script for Unified Source-Filter GAN.

References:
    - https://github.com/bigpon/QPPWG
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import os
import sys
from collections import defaultdict
from logging import getLogger

import hydra
import librosa.display
import matplotlib
import numpy as np
import torch
import usfgan
import usfgan.models
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from usfgan.datasets import AudioFeatDataset
from usfgan.utils.features import SignalGenerator

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


# A logger for this file
logger = getLogger(__name__)


class Trainer(object):
    """Customized trainer module for Unified Source-Filter GAN training."""

    def __init__(
        self,
        config,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            config (dict): Config dict loaded from yaml format configuration file.
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "adversarial", "encode" and "f0" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            device (torch.deive): Pytorch device instance.

        """
        self.config = config
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.finish_train = False
        self.writer = SummaryWriter(config.out_dir)
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config.train.train_max_steps, desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logger.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = {
            "generator": self.model["generator"].state_dict(),
            "discriminator": self.model["discriminator"].state_dict(),
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model["generator"].load_state_dict(state_dict["model"]["generator"])
        self.model["discriminator"].load_state_dict(
            state_dict["model"]["discriminator"]
        )
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            self.optimizer["discriminator"].load_state_dict(
                state_dict["optimizer"]["discriminator"]
            )
            self.scheduler["generator"].load_state_dict(
                state_dict["scheduler"]["generator"]
            )
            self.scheduler["discriminator"].load_state_dict(
                state_dict["scheduler"]["discriminator"]
            )

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        z, c, df, f0 = x
        y_real = y.to(self.device)

        # generator forward
        y_fake, s = self.model["generator"](z, c, df)[:2]

        # calculate spectral loss
        if isinstance(self.criterion["stft"], usfgan.losses.MultiResolutionSTFTLoss):
            # Parallel WaveGAN Multi-Resolution STFT Loss
            sc_loss, mag_loss = self.criterion["stft"](y_fake, y_real)
            gen_loss = self.config.train.lambda_stft * (sc_loss + mag_loss)
            self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
            self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        elif isinstance(
            self.criterion["stft"], usfgan.losses.MultiResolutionLogSTFTPowerLoss
        ):
            # Neural Source-Filter Multi-Resolution STFT Loss
            stft_loss = self.criterion["stft"](y_fake, y_real)
            gen_loss = self.config.train.lambda_stft * stft_loss
            self.total_train_loss["train/log_stft_power_loss"] += stft_loss.item()
        elif isinstance(self.criterion["stft"], usfgan.losses.MelSpectralLoss):
            # HiFiGAN Mel-Spectrogram Reconstruction Loss
            mel_loss = self.criterion["stft"](y_fake, y_real)
            gen_loss = self.config.train.lambda_stft * mel_loss
            self.total_train_loss["train/log_mel_spec_loss"] += mel_loss.item()

        # calculate source regularization loss for usfgan-based models
        if self.config.train.lambda_source > 0:
            if isinstance(
                self.criterion["source"],
                usfgan.losses.ResidualLoss,
            ):
                source_loss = self.criterion["source"](s, y_real, f0)
                gen_loss += self.config.train.lambda_source * source_loss
                self.total_train_loss["train/source_loss"] += source_loss.item()
            else:
                source_loss = self.criterion["source"](s, f0)
                gen_loss += self.config.train.lambda_source * source_loss
                self.total_train_loss["train/source_loss"] += source_loss.item()

        # calculate discriminator related losses
        if self.steps > self.config.train.discriminator_train_start_steps:
            # calculate feature matching loss
            if self.config.train.lambda_feat_match > 0:
                p_fake, fmaps_fake = self.model["discriminator"](
                    y_fake, return_fmaps=True
                )
                with torch.no_grad():
                    p_real, fmaps_real = self.model["discriminator"](
                        y_real, return_fmaps=True
                    )
                fm_loss = self.criterion["feat_match"](fmaps_fake, fmaps_real)
                gen_loss += self.config.train.lambda_feat_match * fm_loss
                self.total_train_loss["train/feat_match_loss"] += fm_loss.item()
            else:
                p_fake = self.model["discriminator"](y_fake)
            # calculate adversarial loss
            adv_loss = self.criterion["adversarial"](p_fake)
            gen_loss += self.config.train.lambda_adv * adv_loss
            self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config.train.generator_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config.train.generator_grad_norm,
            )
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        # discriminator
        if self.steps > self.config.train.discriminator_train_start_steps:
            # re-compute y_fake
            with torch.no_grad():
                y_fake = self.model["generator"](z, c, df)[0]
            # calculate discriminator loss
            p_fake = self.model["discriminator"](y_fake.detach())
            p_real = self.model["discriminator"](y_real)
            # NOTE: the first argument must to be the fake samples
            fake_loss, real_loss = self.criterion["adversarial"](p_fake, p_real)
            dis_loss = fake_loss + real_loss
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config.train.discriminator_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config.train.discriminator_grad_norm,
                )
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logger.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        z, c, df, f0 = x
        y_real = y.to(self.device)

        # generator forward
        y_fake, s = self.model["generator"](z, c, df)[:2]

        # calculate spectral loss
        if isinstance(self.criterion["stft"], usfgan.losses.MultiResolutionSTFTLoss):
            # Parallel WaveGAN Multi-Resolution STFT Loss
            sc_loss, mag_loss = self.criterion["stft"](y_fake, y_real)
            gen_loss = self.config.train.lambda_stft * (sc_loss + mag_loss)
            self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
            self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        elif isinstance(
            self.criterion["stft"], usfgan.losses.MultiResolutionLogSTFTPowerLoss
        ):
            # Neural Source-Filter Multi-Resolution STFT Loss
            stft_loss = self.criterion["stft"](y_fake, y_real)
            gen_loss = self.config.train.lambda_stft * stft_loss
            self.total_eval_loss["eval/log_stft_power_loss"] += stft_loss.item()
        elif isinstance(self.criterion["stft"], usfgan.losses.MelSpectralLoss):
            # HiFiGAN Mel-Spectrogram Reconstruction Loss
            mel_loss = self.criterion["stft"](y_fake, y_real)
            gen_loss = self.config.train.lambda_stft * mel_loss
            self.total_eval_loss["eval/log_mel_spec_loss"] += mel_loss.item()

        # calculate source regularization loss for usfgan-based models
        if self.config.train.lambda_source > 0:
            if isinstance(
                self.criterion["source"],
                usfgan.losses.ResidualLoss,
            ):
                source_loss = self.criterion["source"](s, y_real, f0)
                gen_loss += self.config.train.lambda_source * source_loss
                self.total_eval_loss["eval/source_loss"] += source_loss.item()
            else:
                source_loss = self.criterion["source"](s, f0)
                gen_loss += self.config.train.lambda_source * source_loss
                self.total_eval_loss["eval/source_loss"] += source_loss.item()

        # calculate discriminator related losses
        if self.steps > self.config.train.discriminator_train_start_steps:
            # calculate feature matching loss
            if self.config.train.lambda_feat_match > 0:
                p_fake, fmaps_fake = self.model["discriminator"](
                    y_fake, return_fmaps=True
                )
                p_real, fmaps_real = self.model["discriminator"](
                    y_real, return_fmaps=True
                )
                fm_loss = self.criterion["feat_match"](fmaps_fake, fmaps_real)
                gen_loss += self.config.train.lambda_feat_match * fm_loss
                self.total_eval_loss["eval/feat_match_loss"] += fm_loss.item()
            else:
                p_fake = self.model["discriminator"](y_fake)
            # calculate adversarial loss
            adv_loss = self.criterion["adversarial"](p_fake)
            gen_loss += self.config.train.lambda_adv * adv_loss
            self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()

        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()

        # discriminator
        if self.steps > self.config.train.discriminator_train_start_steps:
            # calculate discriminator loss
            p_real = self.model["discriminator"](y_real)
            # NOTE: the first augment must to be the fake sample
            fake_loss, real_loss = self.criterion["adversarial"](p_fake, p_real)
            dis_loss = fake_loss + real_loss
            self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
            self.total_eval_loss["eval/real_loss"] += real_loss.item()
            self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["valid"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)
            if eval_steps_per_epoch == 3:
                break

        logger.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logger.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        x_batch, y_real_batch = batch
        # use only the first sample
        x_batch = [x[:1].to(self.device) for x in x_batch]
        y_real_batch = y_real_batch[:1]
        z_batch, c_batch, df_batch = x_batch[:3]

        # generator forward
        h_batch, n_batch, a_batch = None, None, None
        if isinstance(
            self.model["generator"],
            (
                usfgan.models.ParallelHnUSFGANGenerator,
                usfgan.models.CascadeHnUSFGANGenerator,
            ),
        ):
            y_fake_batch, s_batch, h_batch, n_batch, a_batch = self.model["generator"](
                z_batch, c_batch, df_batch
            )
        else:
            y_fake_batch, s_batch = self.model["generator"](z_batch, c_batch, df_batch)

        len50ms = int(self.config.data.sample_rate * 0.05)
        start = np.random.randint(0, self.config.data.batch_max_length - len50ms)
        end = start + len50ms

        for audio, name, save_wav in zip(
            [y_real_batch, y_fake_batch, s_batch, h_batch, n_batch],
            ["real", "fake", "source", "harmonic", "noise"],
            [True, True, True, False, False],
        ):
            if audio is not None:
                audio = audio.view(-1).cpu().numpy()

                # plot spectrogram
                fig = plt.figure(figsize=(8, 6))
                spectrogram = np.abs(
                    librosa.stft(
                        y=audio,
                        n_fft=1024,
                        hop_length=128,
                        win_length=1024,
                        window="hann",
                    )
                )
                spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
                librosa.display.specshow(
                    spectrogram_db,
                    sr=self.config.data.sample_rate,
                    y_axis="linear",
                    x_axis="time",
                    hop_length=128,
                )
                self.writer.add_figure(f"spectrogram/{name}", fig, self.steps)
                plt.clf()
                plt.close()

                # plot full waveform
                fig = plt.figure(figsize=(6, 3))
                plt.plot(audio, linewidth=1)
                self.writer.add_figure(f"waveform/{name}", fig, self.steps)
                plt.clf()
                plt.close()

                # plot short term waveform
                fig = plt.figure(figsize=(6, 3))
                plt.plot(audio[start:end], linewidth=1)
                self.writer.add_figure(f"short_waveform/{name}", fig, self.steps)
                plt.clf()
                plt.close()

                # save as wavfile
                if save_wav:
                    audio = audio / np.max(np.abs(audio))
                    self.writer.add_audio(
                        f"audio_{name}.wav",
                        audio,
                        self.steps,
                        self.config.data.sample_rate,
                    )

            # plot aperiodicity weights
            if a_batch is not None:
                fig = plt.figure(figsize=(6, 4))
                plt.imshow(a_batch.squeeze(0).cpu().numpy(), aspect="auto")
                plt.colorbar()
                self.writer.add_figure(f"aperiodicity", fig, self.steps)
                plt.clf()
                plt.close()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config.train.save_interval_steps == 0:
            self.save_checkpoint(
                os.path.join(
                    self.config.out_dir,
                    "checkpoints",
                    f"checkpoint-{self.steps}steps.pkl",
                )
            )
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config.train.eval_interval_steps == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config.train.log_interval_steps == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config.train.log_interval_steps
                logger.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config.train.train_max_steps:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_length=12000,
        sample_rate=24000,
        hop_size=120,
        aux_context_window=2,
        sine_amp=0.1,
        noise_amp=0.003,
        sine_f0_type="contf0",
        signal_types=["sine", "noise"],
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_length (int): The maximum length of batch.
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            sine_amp (float): Amplitude of sine signal.
            noise_amp (float): Amplitude of random noise signal.
            sine_f0_type (str): F0 type for generating sine signal.
            signal_types (list): List of types for input signals.

        """
        if batch_max_length % hop_size != 0:
            batch_max_length += -(batch_max_length % hop_size)
        assert batch_max_length % hop_size == 0
        self.batch_max_length = batch_max_length
        self.batch_max_frames = batch_max_length // hop_size
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.sine_f0_type = sine_f0_type
        self.signal_generator = SignalGenerator(
            sample_rate=sample_rate,
            hop_size=hop_size,
            sine_amp=sine_amp,
            noise_amp=noise_amp,
            signal_types=signal_types,
        )

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise (and sine) batch (B, D, T).
            Tensor: Auxiliary feature batch (B, C, T' + 2 * aux_context_window).
            Tensor: Dilated factor batch (B, 1, T).
            Tensor: F0 sequence batch (B, 1, T').
            Tensor: Target signal batch (B, 1, T).

        """
        # time resolution check
        y_batch, c_batch, df_batch, f0_batch, contf0_batch = [], [], [], [], []
        for idx in range(len(batch)):
            x, c, df, f0, contf0 = batch[idx]
            self._check_length(x, c, df, f0, contf0, 0)
            if len(c) - 2 * self.aux_context_window > self.batch_max_frames:
                # randomly pickup with the batch_max_length length of the part
                interval_start = self.aux_context_window
                interval_end = len(c) - self.batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                y = x[start_step : start_step + self.batch_max_length]
                c = c[
                    start_frame
                    - self.aux_context_window : start_frame
                    + self.aux_context_window
                    + self.batch_max_frames
                ]
                df = df[start_step : start_step + self.batch_max_length]
                f0 = f0[start_frame : start_frame + self.batch_max_frames]
                contf0 = contf0[start_frame : start_frame + self.batch_max_frames]
                self._check_length(
                    y,
                    c,
                    df,
                    f0,
                    contf0,
                    self.aux_context_window,
                )
            else:
                logger.warn(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [y.astype(np.float32).reshape(-1, 1)]  # [(T, 1), ...]
            c_batch += [c.astype(np.float32)]  # [(T' + 2 * aux_context_window, D), ...]
            df_batch += [df.astype(np.float32).reshape(-1, 1)]  # [(T, 1), ...]
            f0_batch += [f0.astype(np.float32).reshape(-1, 1)]  # [(T', 1), ...]
            contf0_batch += [contf0.astype(np.float32).reshape(-1, 1)]  # [(T', 1), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.FloatTensor(np.array(y_batch)).transpose(2, 1)  # (B, 1, T)
        c_batch = torch.FloatTensor(np.array(c_batch)).transpose(
            2, 1
        )  # (B, 1, T' + 2 * aux_context_window)
        df_batch = torch.FloatTensor(np.array(df_batch)).transpose(2, 1)  # (B, 1, T)
        f0_batch = torch.FloatTensor(np.array(f0_batch)).transpose(2, 1)  # (B, 1, T')
        contf0_batch = torch.FloatTensor(np.array(contf0_batch)).transpose(
            2, 1
        )  # (B, 1, T')

        # make input signal batch tensor
        if self.sine_f0_type == "contf0":
            in_batch = self.signal_generator(contf0_batch)
        else:
            in_batch = self.signal_generator(f0_batch)

        return (in_batch, c_batch, df_batch, f0_batch), y_batch

    def _check_length(self, x, c, df, f0, contf0, context_window):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c) - 2 * context_window) * self.hop_size
        assert len(x) == len(df)
        assert len(x) == len(f0) * self.hop_size
        assert len(x) == len(contf0) * self.hop_size


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig) -> None:
    """Run training process."""

    if not torch.cuda.is_available():
        print("CPU")
        device = torch.device("cpu")
    else:
        print("GPU")
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

    # fix seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # check directory existence
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # write config to yaml file
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(OmegaConf.to_yaml(config))

    # get dataset
    if config.data.remove_short_samples:
        feat_length_threshold = (
            config.data.batch_max_length // config.data.hop_size
            + 2 * config.generator.aux_context_window
        )
    else:
        feat_length_threshold = None

    train_dataset = AudioFeatDataset(
        stats=to_absolute_path(config.data.stats),
        audio_list=to_absolute_path(config.data.train_audio),
        feat_list=to_absolute_path(config.data.train_feat),
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        dense_factor=config.data.dense_factor,
        df_f0_type=config.data.df_f0_type,
        aux_feats=config.data.aux_feats,
    )
    logger.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = AudioFeatDataset(
        stats=to_absolute_path(config.data.stats),
        audio_list=to_absolute_path(config.data.valid_audio),
        feat_list=to_absolute_path(config.data.valid_feat),
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        dense_factor=config.data.dense_factor,
        df_f0_type=config.data.df_f0_type,
        aux_feats=config.data.aux_feats,
    )
    logger.info(f"The number of validation files = {len(valid_dataset)}.")

    dataset = {
        "train": train_dataset,
        "valid": valid_dataset,
    }

    # get data loader
    collater = Collater(
        batch_max_length=config.data.batch_max_length,
        aux_context_window=config.generator.aux_context_window,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        sine_amp=config.data.sine_amp,
        noise_amp=config.data.noise_amp,
        sine_f0_type=config.data.sine_f0_type,
        signal_types=config.data.signal_types,
    )
    train_sampler, valid_sampler = None, None

    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collater,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            pin_memory=config.data.pin_memory,
        ),
        "valid": DataLoader(
            dataset=dataset["valid"],
            shuffle=True,
            collate_fn=collater,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=valid_sampler,
            pin_memory=config.data.pin_memory,
        ),
    }

    # define models and optimizers
    model = {
        "generator": hydra.utils.instantiate(config.generator).to(device),
        "discriminator": hydra.utils.instantiate(config.discriminator).to(device),
    }

    # define training criteria
    criterion = {
        "stft": hydra.utils.instantiate(config.train.stft_loss).to(device),
        "adversarial": hydra.utils.instantiate(config.train.adversarial_loss).to(
            device
        ),
    }
    if config.train.lambda_feat_match > 0:
        criterion["feat_match"] = hydra.utils.instantiate(
            config.train.feat_match_loss
        ).to(device)
    if config.train.lambda_source > 0:
        criterion["source"] = hydra.utils.instantiate(config.train.source_loss).to(
            device
        )

    # define optimizers and schedulers
    optimizer = {
        "generator": hydra.utils.instantiate(
            config.train.generator_optimizer,
            params=model["generator"].parameters(),
        ),
        "discriminator": hydra.utils.instantiate(
            config.train.discriminator_optimizer,
            params=model["discriminator"].parameters(),
        ),
    }
    scheduler = {
        "generator": hydra.utils.instantiate(
            config.train.generator_scheduler,
            optimizer=optimizer["generator"],
        ),
        "discriminator": hydra.utils.instantiate(
            config.train.discriminator_scheduler,
            optimizer=optimizer["discriminator"],
        ),
    }

    # define trainer
    trainer = Trainer(
        config=config,
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # load trained parameters from checkpoint
    if config.train.resume:
        resume = os.path.join(
            config.out_dir, "checkpoints", f"checkpoint-{config.train.resume}steps.pkl"
        )
        if os.path.exists(resume):
            trainer.load_checkpoint(resume)
            logger.info(f"Successfully resumed from {resume}.")
        else:
            logger.info(f"Failed to resume from {resume}.")
            sys.exit(0)
    else:
        logger.info("Start a new training process.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(
                config.out_dir, "checkpoints", f"checkpoint-{trainer.steps}steps.pkl"
            )
        )
        logger.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
