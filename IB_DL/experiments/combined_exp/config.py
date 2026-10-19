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


class BigExp(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        super().__init__()
        params.setdefault(
            'prefix', f"model_"+str(params["model"]) + "_optimizer_" +
            str(params["optimizer"]) + "_tf-trength_" + str(params["tf_strength"]) +
            "_img-channels_" + str(params["img_channels"]) + "_seed_" +
            str(params["seed"]) + "_")
        # params.setdefault(
        #     'prefix', f"model_"+str(params["model"]) + "_img-channels_" +
        #     str(params["img_channels"]) + "_fe_layers_" + str(params["fe_layers"]) +
        #     "_fc_layers_" + str(params["fc_layers"]) + "_base_channels_" +
        #     str(params["base_channels"]) + "_seed_" + str(params["seed"]) + "_")
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


# Fig. 14 and Tables 5 and 7 combine two grids, each with the settings logged
# by its published runs:
#  - three-channel grid over seven architectures (result/2026-02-22_23-10-16 and
#    the Vendi runs), line width 3 scaled to each model's input size;
#  - four-channel grid for Models A and B (result/2026-04-24_15-30-57 and
#    result/2026-04-30_13-18-39), line width 2.
GRIDS = [
    {'model': ['flexicnn', 'linearcnn', 'resnet50_pretrained', 'efficientnet_b1',
               'shufflenet-pretrained', 'vgg13-pretrained', 'vit-b-16-pretrained'],
     'img_channels': [['f;abs', 'f;rel', 'v;rel'], ['f;abs', 'v;rel', 'v;abs']],
     'lw': 3, 'num_workers': 32},
    {'model': ['flexicnn', 'linearcnn'],
     'img_channels': [['f;abs', 'f;rel', 'v;abs', 'v;rel']],
     'lw': 2, 'num_workers': 16},
]


def run_experiment(inpt_params: dict, resume: bool = False):

    inpt_params.setdefault('img_size', 128)
    inpt_params.setdefault('epochs', 100)
    inpt_params.setdefault('learning_rate', 0.001)
    inpt_params.setdefault('batch_size', 64)
    inpt_params.setdefault('persistent_workers', True)
    inpt_params.setdefault('optimizer', ["Adam", "SGD"])
    inpt_params.setdefault('tf_strength', [1, 2])
    inpt_params.setdefault('base_channels', 16)
    inpt_params.setdefault('seed', [1, 2, 3, 4, 5])

    combinations = []
    for grid in GRIDS:
        # explicitly passed parameters take precedence over the grid settings
        grid_params = {**grid, **inpt_params}
        keys = list(grid_params.keys())
        values = [v if isinstance(v, list) else [v] for v in grid_params.values()]
        combinations += [dict(zip(keys, c)) for c in product(*values)]

    log_names: List[str | Path] = []
    for params in combinations:
        start_time = time.time()
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
