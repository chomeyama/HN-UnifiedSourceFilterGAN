# Harmonic-plus-Noise Unified Source-Filter GAN implementation with Pytorch


This repo provides official PyTorch implementation of [HN-uSFGAN](https://arxiv.org/abs/2205.06053), a high-fidelity and pitch controllable neural vocoder based on unifid source-filter networks.<br>
HN-uSFGAN is an extended model of [uSFGAN](https://arxiv.org/abs/2104.04668), and this repo includes the original [uSFGAN](https://github.com/chomeyama/UnifiedSourceFilterGAN) implementation with some modifications.<br>

This repository is tested on the following condition.

- Ubuntu 20.04.3 LTS
- Titan RTX 3090 GPU
- Python 3.9.5
- Cuda 11.5
- CuDNN 8.1.1.33-1+cuda11.2

## Environment setup

```bash
$ cd HN-UnifiedSourceFilterGAN
$ pip install -e .
```

Please refer to the [PWG](https://github.com/kan-bayashi/ParallelWaveGAN) repo for more details.

## Folder architecture
- **egs**:
The folder for projects.
- **egs/vctk**:
The folder of the VCTK project example.
- **usfgan**:
The folder of the source codes.

## Run

In this repo, parameters are managed using [Hydra](https://hydra.cc/docs/intro/).<br>
Hydra provides an easy way to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

### Dataset preparation

Create corpus and scp/list files denoting path to the audio according to your own dataset path.<br>
An example of the training scp/list files are provided under `egs/vctk/data/scp/` directory in this repo.

### Preprocessing

```bash
# Move to the project directory
$ cd egs/vctk

# Extract acoustic features (F0, mel-cepstrum, and etc.)
# You can customize parameters according to usfgan/bin/config/extract_features.yaml
$ usfgan-extract-features audio=data/scp/vctk_all_24kHz.scp

# Compute statistics of training and testing data
$ usfgan-compute-statistics feats=data/scp/vctk_train_24kHz.list stats=data/stats/vctk_train_24kHz.joblib
```

### Training

```bash
# Train a model
# Parallel-HN-uSFGAN generator with HiFiGAN discriminator would show best performance
$ usfgan-train generator=parallel_hn_usfgan discriminator=hifigan train=hn_usfgan data=vctk_24kHz out_dir=exp/parallel_hn_usfgan
```

### Inference

```bash
# Decode with natural acoustic features
$ usfgan-decode out_dir=exp/parallel-hn-usfgan/wav/600000steps checkpoint_path=exp/parallel-hn-usfgan/checkpoints/checkpoint-600000steps.pkl
# Decode with halved f0
$ usfgan-decode out_dir=exp/parallel-hn-usfgan/wav/600000steps checkpoint_path=exp/parallel-hn-usfgan/checkpoints/checkpoint-600000steps.pkl f0_factor=0.50
```

### Monitor training progress

```bash
$ tensorboard --logdir exp
```

## Citation
If you find the code is helpful, please cite the following article.

```
@misc{https://doi.org/10.48550/arxiv.2205.06053,
  doi = {10.48550/ARXIV.2205.06053},
  url = {https://arxiv.org/abs/2205.06053},
  author = {Yoneyama, Reo and Wu, Yi-Chiao and Toda, Tomoki},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Authors

Development:
Reo Yoneyama @ Nagoya University ([@chomeyama](https://github.com/chomeyama))<br>
E-mail: `yoneyama.reo@g.sp.m.is.nagoya-u.ac.jp`

Advisors:<br>
Yi-Chiao Wu @ Nagoya University ([@bigpon](https://github.com/bigpon))<br>
E-mail: `yichiao.wu@g.sp.m.is.nagoya-u.ac.jp`<br>
Tomoki Toda @ Nagoya University<br>
E-mail: `tomoki@icts.nagoya-u.ac.jp`
