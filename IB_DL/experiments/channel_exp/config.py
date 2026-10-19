import gc
import itertools
import math
import os
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
import pandas as pd
import torch
from IB_DL import datasets
from IB_DL.config import PathsConfig
from IB_DL.experiments.default_config import DefaultParams
from itertools import product
import time
from IB_DL.utils import cprint


class ChannelExp(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        super().__init__()
        params.setdefault('prefix', f"channels_" + '-'.join(params["transformations"][5]["img_channels"])+"_model_" + str(
            params["model"])+"_seed_" + str(params["seed"])+"_")
        cprint(f"Name: {params['prefix']}", "highlighted_cyan")
        if "experiment_date" in params.keys():
            params.setdefault('output_dir', str(
                Path(os.path.dirname(os.path.realpath(__file__))) / "result" / params["experiment_date"]))
        else:
            params.setdefault('output_dir', str(
                Path(os.path.dirname(os.path.realpath(__file__))) / "result"))
        if params is not None:
            self.set_params(params)
        os.makedirs(self.output_dir, exist_ok=True)


CHANNELS = [
    ['f;abs'],
    ['f;rel'],
    ['v;rel'],
    ['v;abs'],
    ['f;abs', 'f;rel'],
    ['f;abs', 'v;rel'],
    ['f;abs', 'v;abs'],
    ['f;rel', 'v;rel'],
    ['f;rel', 'v;abs'],
    ['v;rel', 'v;abs'],
    ['f;abs', 'f;rel', 'v;rel'],
    ['f;abs', 'f;rel', 'v;abs'],
    ['f;abs', 'v;rel', 'v;abs'],
    ['f;rel', 'v;rel', 'v;abs'],
    ['f;abs', 'f;rel', 'v;rel', 'v;abs']
]

# Table 4 columns, with the settings logged by the published runs:
# result/2026-02-12_16-32-07 (Model B), result/2026-02-16_09-42-07 (Model A),
# result/2026-02-17_00-20-43 (ResNet-50, 3-channel rows) and
# result/2026-06-30_06-13-14 (ResNet-50 with conv1 replaced to take the
# 1-, 2- and 4-channel rows).
ARMS = {
    'linearcnn': {'img_size': 128, 'fe_layers': 6, 'fc_layers': 5,
                  'growth_factor': 1.5, 'bottleneck_channels': 16},
    'flexicnn': {'img_size': 128, 'fe_layers': 4, 'fc_layers': 4, 'base_channels': 16},
    'resnet50_pretrained': {'img_size': 224},
    'modified_resnet50_pretrained': {'img_size': 224},
}


def _channels_for(model: str) -> list[list[str]]:
    if model == 'resnet50_pretrained':
        return [c for c in CHANNELS if len(c) == 3]
    if model == 'modified_resnet50_pretrained':
        return [c for c in CHANNELS if len(c) != 3]
    return CHANNELS


def _transformations(img_size: int, lw: int, img_channels: list[str]) -> list[dict]:
    return [
        {"name": "LoopToTensor"},
        {"name": "AWG", "max_mu": 0.15, "sigma": 0.2, "p": 0.75},
        {"name": "SaltAndPepperNoise", "salt_prob": 0.03,
         "pepper_prob": 0.03, "max_amplitude": 1.0, "p": 0.75},
        {"name": "StretchLoops", "max_delta_r": 0.3,
         "max_num_of_axes": 2, "p": 0.75},
        {"name": "ShiftLoops", "max_shift": 0.1,
         "max_num_of_axes": 2, "p": 0.75},
        {"name": "ToImage", "img_size": img_size, "linewidth": lw,
         "img_channels": img_channels, "num_skips": 3, "max_len_skips": 8, "change_lwidth": False},
        {"name": "RandomRotation", "degree": 8, "p": 0.7},
        {"name": "GaussianBlur", "kernel_size": (
            3, 3), "sigma": (0.1, 1), "p": 0.5}
    ]


def run_experiment(inpt_params: dict, resume: bool = False):

    inpt_params.setdefault('lw', 3)
    inpt_params.setdefault('epochs', 100)
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('learning_rate', 0.001)
    inpt_params.setdefault('batch_size', 64)
    inpt_params.setdefault('num_workers', 32)
    inpt_params.setdefault('persistent_workers', True)
    inpt_params.setdefault('model', list(ARMS))
    inpt_params.setdefault('seed', [1, 2, 3, 4, 5])

    models = inpt_params.pop('model')
    models = models if isinstance(models, list) else [models]
    log_names: List[str | Path] = []
    for model in models:
        # explicitly passed parameters take precedence over the arm settings
        arm_params = {**ARMS.get(model, {}), **inpt_params, 'model': model}
        arm_params.setdefault('transformations', [
            _transformations(arm_params['img_size'], arm_params['lw'], channels)
            for channels in _channels_for(model)])
        log_names += _run_grid(arm_params, resume)

    cprint("Experiment completed. Log files:")
    for log_name in log_names:
        cprint(str(log_name), "highlighted_green")
    return log_names


def _run_grid(inpt_params: dict, resume: bool) -> List[str | Path]:
    keys = list(inpt_params.keys())
    values = [v if isinstance(v, list) else [v] for v in inpt_params.values()]

    log_names: List[str | Path] = []
    for combination in product(*values):
        start_time = time.time()
        params = dict(zip(keys, combination))

        cprint(str(params), "underline_blue")
        object = ChannelExp(params=params)
        files = os.listdir(object.output_dir)
        if any(object.prefix in f for f in files):
            if not any(object.prefix in f and "_model_backup_" in f for f in files):
                cprint(
                    f"Experiment with prefix {object.prefix} already exists. Skipping.", "red")
                continue
            else:
                cprint(
                    f"Experiment with prefix {object.prefix} already exists but possably interrupted. Resuming.", "yellow")
        object.run(resume=resume, example_img=True)
        log_names.append(object.log_filename)
        del object
        torch.cuda.empty_cache()
        gc.collect()

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Total runtime: {elapsed:.2f} seconds")
    return log_names
