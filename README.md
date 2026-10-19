This repository contains the official PyTorch implementation for the paper **"Classification of Intra-Breath Oscillometry Patterns with Deep Learning Methods"** by Á. Sára Németh and György Kalmár.

This framework introduces a novel, end-to-end deep learning pipeline that transforms high-resolution, one-dimensional temporal intra-breath oscillometry (IBOsc) signals into two-dimensional image representations. By framing physiological signal analysis as a computer vision task, we classify respiratory mechanics into three primary clinical groups: **Healthy**, **Chronic Obstructive Pulmonary Disease (COPD)**, and **Interstitial Lung Disease (ILD)**.

## Overview

Traditional analysis of IBOsc data relies heavily on classical machine learning and manual, expert-driven feature engineering. This repository bypasses those limitations by utilizing Convolutional Neural Networks (CNNs) to autonomously discover and extract complex nonlinear mechanical patterns.

**Key Highlights:**

* **Signal-to-Image Rendering:** Converts dynamic resistance ($R_{rs}$) and reactance ($X_{rs}$) trajectories into multi-channel 2D images plotted against flow ($\dot{V}$) and volume ($V$).
* **Physics-Inspired Augmentation:** Custom data augmentation techniques (e.g., signal stretching, shifting, AWGN, and signal dropout) applied directly to the raw temporal signals before image rendering to simulate natural breathing variations and sensor noise.
* **Transfer Learning on Shapes:** Utilizes custom pre-training on a subset of the *Quick, Draw!* dataset to overcome CNN texture bias and force the models to prioritize curved geometric trajectories.
* **Patient-Level Aggregation:** Implements loop-wise classification, median loop evaluation, and soft majority voting (which achieved the best diagnostic stability).
* **Diagnostic Performance:** The optimized pipeline achieved an overall accuracy of 83.1%.

## Repository Structure

* `datasets.py`: Contains the `Loop` and `IBDataset` classes. Handles parsing of `.csv` files and the patient-wise train/validation/test split.
* `transformations.py`: The core preprocessing engine. Contains the `ToImage` transform that plots the multi-channel 2D canvases, as well as both temporal (Pre-render) and spatial (Post-render) augmentation classes.
* `models.py`: A flexible model factory. Includes standard architectures (ResNet, EfficientNet, ViT) alongside our custom, highly parametric CNNs (`Model A` = `flexicnn`, `Model B` = `linearcnn`) optimized for sparse IBOsc images.
* `train.py`: The training orchestrator. Includes the `train_loop`, custom string-label Cross Entropy Loss, learning rate schedulers, early stopping, and TensorBoard/Logger integration.
* `eval.py`: Test-set evaluation with the three aggregation methods (loop-wise, soft vote, median loop); writes `test_results.csv` next to the run logs.
* `visualize.py`: Plotting functions for the manuscript figures.
* `IBOsc/`: Handcrafted intra-breath features of the classical pipeline [8], used for the FCNN baseline of Table 6. This package is not included in the repository; only `features_exp` (Table 6) and Fig. 2 need it, and they report a clear error without it.
* `SIN_QD/`: Stylized-ImageNet model loading and the *Quick, Draw!* pre-training (`quick_draw.py`).
* `experiments/<name>/config.py`: One configuration per experiment of the paper (see below), together with the scripts that turn its results into the tables and figures.

## Dataset

The clinical IBOsc dataset utilized in this study is publicly available.

* **Download:** [Figshare (DOI: 10.6084/m9.figshare.29589464.v2)](https://figshare.com/articles/dataset/Intra-breath_measurements_-_supporting_information_of_Pattern_recognition_in_intra-breath_oscillometry_measurements_/29589464).

The dataset contains measurements from 872 patients, yielding over 39,000 individual breath cycles. The measurements should be transformed into .csv files with the given header:

* `id`: Patient ID
* `index`: Measurement index - loop index. E.g.: 0-10 is the first measurement's 11th loop.
* `label`: Label: 0 for Healthy, 1 for COPD and 2 for ILD.
* `volume_0` to `volume_49`: Volume axis resampled to 50 points.
* `flow_0` to `flow_49`: Flow axis resampled to 50 points.
* `resistance_0` to `resistance_49`: Resistance axis resampled to 50 points.
* `reactance_0` to `reactance_49`: Reactance axis resampled to 50 points.

## Installation & Requirements

1. Clone the repository:

```bash
git clone https://github.com/NemethAgnesSara/Intra-breath_DL.git
cd Intra-breath_DL

```

2. Install the required dependencies:
This project was built using PyTorch. Ensure you have a CUDA-capable environment if you intend to train on a GPU.

```bash
pip install -e .

```

Place the three CSV files in `IB_DL/data/` as `all_measurements_1.csv`, `all_measurements_2.csv` and `all_measurements_3.csv`, and run every command from the repository root (the paths are configured in `config.py`).

## Usage Example

You can run an analysis by using the Click command:

```bash
IB_DL default-experiment --name <name of experiment>

```

This will automatically run and evaluate the experiment. Logs, checkpoints and `test_results.csv` are written to `IB_DL/experiments/<name>/result/<date>/`. Any default of the experiment can be overridden on the command line, e.g. `--seed 1 --epochs 50`.

## Reproducing the Paper

**Quick, Draw! pre-training (Table 3).** `python -m IB_DL.SIN_QD.quick_draw` downloads the ten classes into `IB_DL/data/QuickDraw/` and trains the two ResNet-50 checkpoints (random and ImageNet initialisation) that `pre_train` loads from `IB_DL/SIN_QD/result/`. The published checkpoints were trained on an earlier download of the same dataset that is not byte-identical with the current files.

**Reproducibility.** Training is deterministic for a given seed: data splitting, sampling, augmentation and initialisation are seeded, and repeating a run reproduces its log exactly. Augmentation is drawn inside the data-loader workers, so the same numbers also require the same `num_workers`; the defaults use the value of the published runs. All experiments were ran using seeds 1 to 5. The published runs were made on an NVIDIA GeForce RTX 3090; exact reproduction was verified with PyTorch 2.7.1.

## Citation

If you use this code or the associated dataset in your research, please cite our paper:

```bibtex
@article{nemeth2026ibosc,
  title={Classification of Intra-Breath Oscillometry Patterns with Deep Learning Methods},
  author={N{\'e}meth, {\'A}. S{\'a}ra and Kalm{\'a}r, Gy{\"o}rgy},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}

```

## Contact

For any questions or issues regarding the code, please open an issue in this repository or contact Á. Sára Németh at `nemetha@inf.u-szeged.hu`.