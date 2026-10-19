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


class LineWidth(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        super().__init__()
        params.setdefault('prefix', f"linewidth_" + str(params["lw"])+"_model_" + str(
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


def run_experiment(inpt_params: dict, resume: bool = False):

    inpt_params.setdefault('img_size', 0)
    inpt_params.setdefault('lw', [1, 2, 3, 4, 6, 8, 10])
    inpt_params.setdefault('epochs', 100)
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('learning_rate', 0.001)
    inpt_params.setdefault('batch_size', 64)
    inpt_params.setdefault('num_workers', 32)
    inpt_params.setdefault('persistent_workers', True)

    # inpt_params.setdefault('model', ['resnet50_pretrained', 'flexicnn', 'efficientnet_b1', 'linearcnn'])
    inpt_params.setdefault(
        'model', ['resnet50_pretrained', 'flexicnn', 'efficientnet_b1', 'linearcnn'])

    # inpt_params.setdefault('model', 'flexicnn')

    inpt_params.setdefault('seed', [1, 2, 3, 4, 5])

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
        elif params['model'] == 'resnet50_pretrained':
            params['img_size'] = 224
            params['lw'] = math.floor(params['img_size'] * params['lw']/128)
        elif params['model'] == 'efficientnet_b1':
            params['img_size'] = 240
            params['lw'] = math.floor(params['img_size'] * params['lw']/128)
        if params['lw'] < 1:
            params['lw'] = 1

        cprint(str(params), "underline_blue")
        object = LineWidth(params=params)
        files = os.listdir(object.output_dir)
        if any(object.prefix in f for f in files):
            if not any(object.prefix in f and "_model_backup_" in f for f in files):
                cprint(
                    f"Experiment with prefix {object.prefix} already exists. Skipping.", "red")
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


def sweep_linewidts(resolution, img_size_list):
    """
    Create a grid of images for different line widths and image sizes.

    :param line_width_list: list of line widths to test
    :param img_size_list: list of image sizes to test
    """
    combination = itertools.product(resolution, img_size_list)
    samples = []
    lws = []
    img_sizes = []
    resolutions = []
    for res, img_size in combination:
        lw = math.floor(img_size * res)
        params = {
            'seed': 1,
            'img_size': img_size,
            'lw': lw,
            'experiment_date': 'local_debug',
            'all_measurements': True,
            'data_dir': str(PathsConfig.data_dir),
            'num_rows': 100,
            'test_aggregation': 'all',
            'val_aggregation': 'none',
            'persistent_workers': False,
            'prefetch_factor': 1,
            'num_workers': 1,
            'batch_size': 1,
            'class_labels': None,
            'ttv_ratio': [0.6, 0.2, 0.2],
            'transformations': [{"name": "LoopToTensor"}, {'name': 'ToImage', 'img_size': img_size, 'linewidth': lw, 'img_channels': ['f;abs', 'f;rel', 'v;abs']}],
        }
        _, _, _, ib_dataset = datasets.init_dataset(params)
        sample, label = ib_dataset[0]
        samples.append(sample)
        lws.append(lw)
        resolutions.append(res)
        img_sizes.append(img_size)

    samples_df = pd.DataFrame(
        {'sample': samples, 'line_width': lws, 'img_size': img_sizes, 'resolution': resolutions})
    fig, axes = plt.subplots(len(img_size_list)*2, len(resolution),
                             figsize=(len(resolution)*4, len(img_size_list)*4*2))
    for idx, row in samples_df.iterrows():
        i = samples_df['img_size'].unique().tolist().index(row['img_size'])
        j = samples_df['resolution'].unique().tolist().index(row['resolution'])
        axes[i, j].imshow(row['sample'][0], cmap='gray')
        axes[i, j].set_title(
            f'Img Size: {row["img_size"]}, LW: {row["line_width"]}, Res: {row["resolution"]}')
        axes[i, j].axis('off')
        axes[i+len(img_size_list), j].imshow(row['sample'][1], cmap='gray')
        axes[i+len(img_size_list),
             j].set_title(f'Img Size: {row["img_size"]}, LW: {row["line_width"]}')
        axes[i+len(img_size_list), j].axis('off')
    plt.tight_layout()
    plt.savefig("IB_DL/images/img_size-lw/img_size_3.png")

    fig, axes = plt.subplots(len(img_size_list), len(resolution),
                             figsize=(len(resolution)*2, len(img_size_list)*2))
    for idx, row in samples_df.iterrows():
        i = samples_df['img_size'].unique().tolist().index(row['img_size'])
        j = samples_df['resolution'].unique().tolist().index(row['resolution'])
        axes[i, j].imshow(row['sample'][1], cmap='gray')
        # axes[i, j].set_title(
        #     f'Img Size: {row["img_size"]}, LW: {row["line_width"]}, Res: {row["resolution"]}')
        axes[i, j].axis('off')

        # Add axis labels
        for j in range(len(resolution)):
            axes[len(img_size_list)-1,
                 j].set_xlabel(f"{int(resolution[j]*128)}/128", fontsize=14)
            axes[len(img_size_list)-1, j].axis('on')
            axes[len(img_size_list)-1, j].set_xticks([])
            axes[len(img_size_list)-1, j].set_yticks([])

        for i in range(len(img_size_list)):
            axes[i, 0].set_ylabel(f"{img_sizes[i]}", fontsize=14)
            axes[i, 0].axis('on')
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

    plt.tight_layout()
    plt.savefig("IB_DL/images/img_size-lw/img_size_4.pdf")

    # python -c "from IB_DL.experiments.linewidth_exp.config import sweep_linewidts; sweep_linewidts([1/128, 2/128, 3/128, 4/128, 6/128, 8/128], [128,224, 256])"
