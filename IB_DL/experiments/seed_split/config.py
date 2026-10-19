"""Initialisation variance on a fixed patient split, plus a seed ensemble.

The 2026-08-31 seed experiment could not separate the two sources of seed
variance: one seed drove both the patient split and the weight initialisation,
and a temporary edit that pinned the split was never committed, so evaluation
rebuilt a different split for each run. This experiment does it properly with
``split_seed``: every run trains and is scored on the same split (seed 5), and
only the training seed - initialisation, sampler order, augmentation draws -
varies from 1 to 5.

Run ``ensemble.py`` on the result directory afterwards to combine the five
models by probability averaging.

The defaults reproduce Section VI-K: Model B on each of the five patient splits
(split_seed 1-5), five training seeds per split.

Settings follow the manuscript's pipeline - Adam, flat rendering, mild
augmentation, early stopping - rather than the original run, which used SGD and
(through the old gradient default) rendered with the temporal ramp.
"""

import gc
import os
from pathlib import Path
from typing import List

import torch
from IB_DL.experiments.default_config import DefaultParams
from itertools import product
import time
from IB_DL.utils import cprint

SPLIT_SEEDS = [1, 2, 3, 4, 5]
TRAIN_SEEDS = [1, 2, 3, 4, 5]

# Manuscript standard. The original run used SGD and gradient rendering; set
# these to "SGD" / True to replicate it instead.
OPTIMIZER = "Adam"
GRADIENT = False


class SeedSplitExp(DefaultParams):
    def __init__(self, params: dict):
        super().__init__()
        cprint(f"Name: {params['prefix']}", "highlighted_cyan")
        if "experiment_date" in params.keys():
            params.setdefault('output_dir', str(Path(os.path.dirname(
                os.path.realpath(__file__))) / "result" / params["experiment_date"]))
        else:
            params.setdefault('output_dir', str(
                Path(os.path.dirname(os.path.realpath(__file__))) / "result"))
        self.set_params(params)
        os.makedirs(self.output_dir, exist_ok=True)


def _transformations(img_size: int, lw: int, img_channels: list,
                     gradient: bool) -> list[dict]:
    """tf_strength=1 pipeline of default_config, with the rendering pinned."""
    return [
        {"name": "LoopToTensor"},
        {"name": "AWG", "max_mu": 0.1, "sigma": 0.1, "p": 0.25},
        {"name": "SaltAndPepperNoise", "salt_prob": 0.01,
            "pepper_prob": 0.01, "max_amplitude": 1.0, "p": 0.25},
        {"name": "StretchLoops", "max_delta_r": 0.15,
            "max_num_of_axes": 2, "p": 0.25},
        {"name": "ShiftLoops", "max_shift": 0.05,
            "max_num_of_axes": 2, "p": 0.25},
        {"name": "ToImage", "img_size": img_size, "linewidth": lw,
            "img_channels": img_channels, "num_skips": 2, "max_len_skips": 4,
            "gradient": gradient},
        {"name": "RandomRotation", "degree": 5, "p": 0.2},
        {"name": "GaussianBlur", "kernel_size": (
            3, 3), "sigma": (0.05, 0.1), "p": 0.2}
    ]


def run_experiment(inpt_params: dict, resume: bool = False):
    inpt_params.setdefault('model', 'linearcnn')
    inpt_params.setdefault('split_seed', SPLIT_SEEDS)
    inpt_params.setdefault('seed', TRAIN_SEEDS)

    # --- manuscript pipeline --------------------------------------------------
    inpt_params.setdefault('optimizer', OPTIMIZER)
    inpt_params.setdefault('epochs', 100)      # early stopping decides
    inpt_params.setdefault('tf_strength', 1)
    inpt_params.setdefault('img_channels', [['f;abs', 'f;rel', 'v;rel']])
    inpt_params.setdefault('img_size', 224)
    inpt_params.setdefault('lw', 3)
    inpt_params.setdefault('learning_rate', 0.001)
    inpt_params.setdefault('batch_size', 64)
    inpt_params.setdefault('num_workers', 16)
    inpt_params.setdefault('persistent_workers', True)

    keys = list(inpt_params.keys())
    values = [v if isinstance(v, list) else [v] for v in inpt_params.values()]

    log_names: List[str | Path] = []
    for combination in product(*values):
        start_time = time.time()
        params = dict(zip(keys, combination))
        # Custom CNNs run at 128 px with the layouts used everywhere else in
        # the paper; the 224 px / lw 3 defaults above are for ResNet-50.
        if params['model'] == 'linearcnn':        # Model B, lw as in combined_exp
            params.update(img_size=128, lw=3, fe_layers=6, fc_layers=5,
                          growth_factor=1.5, bottleneck_channels=16)
        elif params['model'] == 'flexicnn':       # Model A
            params.update(img_size=128, lw=2, fe_layers=4, fc_layers=4,
                          base_channels=16)
        params['transformations'] = _transformations(
            params['img_size'], params['lw'], params['img_channels'], GRADIENT)
        params['prefix'] = (
            "model_" + str(params["model"]) + "_optimizer_" +
            str(params["optimizer"]) + "_split-seed_" + str(params["split_seed"]) +
            "_seed_" + str(params["seed"]) + "_")

        cprint(str(params), "underline_blue")
        object = SeedSplitExp(params=params)
        files = os.listdir(object.output_dir)
        if any(object.prefix in f for f in files):
            if not any(object.prefix in f and "_model_backup_" in f for f in files):
                cprint(
                    f"Experiment with prefix {object.prefix} already exists. Skipping.", "red")
                del object
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                cprint(
                    f"Experiment with prefix {object.prefix} already exists but possably interrupted. Resuming.", "yellow")
        object.run(resume=resume)
        log_names.append(object.log_filename)
        del object
        torch.cuda.empty_cache()
        gc.collect()

        end_time = time.time()
        print(f"Total runtime: {end_time - start_time:.2f} seconds")
    cprint("Experiment completed. Log files:")
    for log_name in log_names:
        cprint(str(log_name), "highlighted_green")
    return log_names
