"""Temporal-encoding A/B: the same configuration rendered with and without the
gradient (time-encoded line intensity) of ``ToImage``.

Answers R3's "implement a temporal encoding strategy to demonstrate its effect
on accuracy". Only ``gradient`` varies; everything else is pinned to what the
earlier sections already settled:

    img_channels  ['f;abs', 'v;rel', 'v;abs']  three-channel standard (VI-H)
    tf_strength   1                            mild augmentation (Table 5)
    optimizer     Adam                         the stable/reproducible choice
    img_size/lw   128/2, pretrained 224/3      as combined_exp (VI-G: no effect)

Varied: architecture x {gradient on, off} over 5 seeds = 30 runs. The three
architectures are the ones that recur across the paper's other ablations
(custom high-capacity, custom bottlenecked, standard pretrained baseline).

The previous grid-search version of this file is kept as
``config_2026-06_grid.py.bak``; its results live in ``result/2026-06-29_*``.
"""

import gc
import math
import os
from pathlib import Path
from typing import List

import torch
from IB_DL.experiments.default_config import DefaultParams
from itertools import product
import time
from IB_DL.utils import cprint


class BigExp(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        super().__init__()
        cprint(f"Name: {params['prefix']}", "highlighted_cyan")
        if "experiment_date" in params.keys():
            params.setdefault('output_dir', str(Path(os.path.dirname(
                os.path.realpath(__file__))) / "result" / params["experiment_date"]))
        else:
            params.setdefault('output_dir', str(
                Path(os.path.dirname(os.path.realpath(__file__))) / "result"))
        if params is not None:
            self.set_params(params)
        os.makedirs(self.output_dir, exist_ok=True)


def _transformations(img_size: int, lw: int, img_channels: list,
                     gradient: bool) -> list[dict]:
    """tf_strength=1 pipeline of default_config, with the rendering pinned.

    ``gradient`` is written explicitly so the arm ends up in the run log; it is
    the one thing this experiment varies and must never fall back to a default.
    """
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

    # --- the only variable ---------------------------------------------------
    inpt_params.setdefault('gradient', [False, True])

    # --- held fixed by the earlier sections -----------------------------------
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('tf_strength', 1)
    # wrapped in an outer list: product() would otherwise iterate the channels
    inpt_params.setdefault('img_channels', [['f;abs', 'v;rel', 'v;abs']])
    inpt_params.setdefault('img_size', 128)
    inpt_params.setdefault('lw', 2)
    inpt_params.setdefault('base_channels', 16)

    inpt_params.setdefault(
        'model', ['flexicnn', 'linearcnn', 'resnet50_pretrained'])
    inpt_params.setdefault('seed', [1, 2, 3, 4, 5])

    # --- training setup -------------------------------------------------------
    inpt_params.setdefault('epochs', 100)
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
        if params['model'] == 'flexicnn':
            print("Setting default params for flexicnn")
            params['img_size'] = 128
            params['fe_layers'] = 4
            params['fc_layers'] = 4
            params['base_channels'] = 16
            params['lw'] = math.floor(params['img_size'] * params['lw']/128)
        elif params['model'] == 'linearcnn':
            params['img_size'] = 128
            params['fe_layers'] = 6
            params['fc_layers'] = 5
            params['growth_factor'] = 1.5
            params['bottleneck_channels'] = 16
            params['lw'] = math.floor(params['img_size'] * params['lw']/128)
        elif 'resnet' in params['model'] or 'vgg13' in params['model'] or 'vit' in params['model'] or 'shufflenet' in params['model']:
            params['img_size'] = 224
            params['lw'] = math.floor(params['img_size'] * params['lw']/128)
        elif 'inception' in params['model']:
            params['img_size'] = 299
            params['lw'] = math.floor(params['img_size'] * params['lw']/128)
        elif 'efficientnet_b1' in params['model']:
            params['img_size'] = 240
            params['lw'] = math.floor(params['img_size'] * params['lw']/128)
        if params['lw'] < 1:
            params['lw'] = 1

        # Built here, not before the loop: the block above has just set the
        # per-model img_size/lw, and the pipeline has to use those.
        gradient = params.pop('gradient')
        params['transformations'] = _transformations(
            params['img_size'], params['lw'], params['img_channels'], gradient)

        # The arm goes in the prefix: the "already exists" guard below is
        # prefix-keyed, model_label in test_results.csv is the prefix, and the
        # two arms would otherwise overwrite each other's checkpoints.
        params['prefix'] = (
            "model_" + str(params["model"]) + "_optimizer_" +
            str(params["optimizer"]) + "_tf-trength_" + str(params["tf_strength"]) +
            "_img-channels_" + str(params["img_channels"]) +
            "_gradient_" + str(gradient) + "_seed_" + str(params["seed"]) + "_")

        cprint(str(params), "underline_blue")
        object = BigExp(params=params)
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
        elapsed = end_time - start_time
        print(f"Total runtime: {elapsed:.2f} seconds")
    cprint("Experiment completed. Log files:")
    for log_name in log_names:
        cprint(str(log_name), "highlighted_green")
    return log_names
