import gc
import os
from pathlib import Path
from typing import List, Optional
from unittest import result
from IB_DL.config import _REPO_ROOT, PathsConfig
from IB_DL.models import get_model
from IB_DL.datasets import init_dataset
from matplotlib import pyplot as plt
from IB_DL.experiments.default_config import DefaultParams
from IB_DL.logger import LogReader
from datetime import date, datetime
import torch
from pytorch_grad_cam import GradCAM, EigenCAM, ScoreCAM, AblationCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from itertools import product
from IB_DL import eval
import time
from IB_DL.utils import check_param_compatibility, cprint, init_logreaders, set_seed


class ArchExp(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        super().__init__()
        params.setdefault('prefix', f"fe_layers_" + str(params["fe_layers"])+"_fc_layers_" + str(
            params["fc_layers"])+"_rnd_" + str(params["seed"])+"_")
        cprint(f"Name: {params['prefix']}", "highlighted_cyan")
        # universal path configuration
        global_store = PathsConfig.output_dir
        current_script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        try:
            relative_path = current_script_dir.relative_to(_REPO_ROOT)
            print("Relative path:", relative_path)
        except ValueError:
            relative_path = Path("custom_experiment")
        base_output = global_store / relative_path / \
            "result"  # global_store / relative_path / "result"
        if "experiment_date" in params:
            final_output_path = base_output / params["experiment_date"]
        else:
            final_output_path = base_output
        # The prefix does not name the model, so each architecture gets its own
        # directory (as Model A and Model B had in the published runs).
        final_output_path = final_output_path / params["model"]
        params['output_dir'] = str(final_output_path)

        if params is not None:
            self.set_params(params)
        os.makedirs(self.output_dir, exist_ok=True)
        self.fe_layers = params["fe_layers"]
        self.fc_layers = params["fc_layers"]
        self.base_channels = params.get("base_channels", 1)


def run_experiment(inpt_params: dict, resume: bool = False):

    # missing_configs = [
    # {'fe_layers': 1, 'fc_layers': 1, 'seed': 1},
    # {'fe_layers': 1, 'fc_layers': 1, 'seed': 5},
    # {'fe_layers': 1, 'fc_layers': 2, 'seed': 1},
    # {'fe_layers': 1, 'fc_layers': 2, 'seed': 5},
    # {'fe_layers': 1, 'fc_layers': 3, 'seed': 1},
    # {'fe_layers': 1, 'fc_layers': 3, 'seed': 5},
    # {'fe_layers': 1, 'fc_layers': 4, 'seed': 1},
    # {'fe_layers': 1, 'fc_layers': 4, 'seed': 5},
    # {'fe_layers': 1, 'fc_layers': 5, 'seed': 1},
    # {'fe_layers': 1, 'fc_layers': 5, 'seed': 5},
    # {'fe_layers': 1, 'fc_layers': 6, 'seed': 1},
    # {'fe_layers': 1, 'fc_layers': 6, 'seed': 5},
    # {'fe_layers': 1, 'fc_layers': 7, 'seed': 1},
    # {'fe_layers': 1, 'fc_layers': 7, 'seed': 5},
    # {'fe_layers': 1, 'fc_layers': 8, 'seed': 1},
    # {'fe_layers': 1, 'fc_layers': 8, 'seed': 5},
    # {'fe_layers': 2, 'fc_layers': 1, 'seed': 1},
    # {'fe_layers': 2, 'fc_layers': 1, 'seed': 5},
    # {'fe_layers': 2, 'fc_layers': 2, 'seed': 1},
    # {'fe_layers': 2, 'fc_layers': 2, 'seed': 5},
    # {'fe_layers': 2, 'fc_layers': 3, 'seed': 1},
    # {'fe_layers': 2, 'fc_layers': 3, 'seed': 5},
    # {'fe_layers': 2, 'fc_layers': 4, 'seed': 1},
    # {'fe_layers': 2, 'fc_layers': 4, 'seed': 5},
    # {'fe_layers': 2, 'fc_layers': 5, 'seed': 1},
    # {'fe_layers': 2, 'fc_layers': 5, 'seed': 5},
    # {'fe_layers': 2, 'fc_layers': 6, 'seed': 1},
    # {'fe_layers': 2, 'fc_layers': 6, 'seed': 5},
    # {'fe_layers': 2, 'fc_layers': 7, 'seed': 1},
    # {'fe_layers': 2, 'fc_layers': 7, 'seed': 5},
    # {'fe_layers': 2, 'fc_layers': 8, 'seed': 1},
    # {'fe_layers': 2, 'fc_layers': 8, 'seed': 5},
    # {'fe_layers': 3, 'fc_layers': 1, 'seed': 1},
    # {'fe_layers': 3, 'fc_layers': 1, 'seed': 5},
    # {'fe_layers': 3, 'fc_layers': 2, 'seed': 1},
    # {'fe_layers': 3, 'fc_layers': 2, 'seed': 5},
    # {'fe_layers': 3, 'fc_layers': 3, 'seed': 1},
    # {'fe_layers': 3, 'fc_layers': 3, 'seed': 5},
    # {'fe_layers': 3, 'fc_layers': 4, 'seed': 1},
    # {'fe_layers': 3, 'fc_layers': 4, 'seed': 5},
    # {'fe_layers': 3, 'fc_layers': 5, 'seed': 1},
    # {'fe_layers': 3, 'fc_layers': 5, 'seed': 5},
    # {'fe_layers': 3, 'fc_layers': 6, 'seed': 1},
    # {'fe_layers': 3, 'fc_layers': 6, 'seed': 5},
    # {'fe_layers': 3, 'fc_layers': 7, 'seed': 1},
    # {'fe_layers': 3, 'fc_layers': 7, 'seed': 5},
    # {'fe_layers': 3, 'fc_layers': 8, 'seed': 1},
    # {'fe_layers': 3, 'fc_layers': 8, 'seed': 5},
    # {'fe_layers': 4, 'fc_layers': 1, 'seed': 1},
    # {'fe_layers': 4, 'fc_layers': 1, 'seed': 5},
    # {'fe_layers': 4, 'fc_layers': 2, 'seed': 1},
    # {'fe_layers': 4, 'fc_layers': 2, 'seed': 5},
    # {'fe_layers': 4, 'fc_layers': 3, 'seed': 1},
    # {'fe_layers': 4, 'fc_layers': 3, 'seed': 5},
    # {'fe_layers': 4, 'fc_layers': 4, 'seed': 1},
    # {'fe_layers': 4, 'fc_layers': 4, 'seed': 5},
    # {'fe_layers': 4, 'fc_layers': 5, 'seed': 1},
    # {'fe_layers': 4, 'fc_layers': 5, 'seed': 4},
    # {'fe_layers': 4, 'fc_layers': 5, 'seed': 5},
    # {'fe_layers': 4, 'fc_layers': 6, 'seed': 1},
    # {'fe_layers': 4, 'fc_layers': 6, 'seed': 4},
    # {'fe_layers': 4, 'fc_layers': 6, 'seed': 5},
    # {'fe_layers': 4, 'fc_layers': 7, 'seed': 1},
    # {'fe_layers': 4, 'fc_layers': 7, 'seed': 4},
    # {'fe_layers': 4, 'fc_layers': 7, 'seed': 5},
    # {'fe_layers': 4, 'fc_layers': 8, 'seed': 1},
    # {'fe_layers': 4, 'fc_layers': 8, 'seed': 4},
    # {'fe_layers': 4, 'fc_layers': 8, 'seed': 5},
    # {'fe_layers': 5, 'fc_layers': 1, 'seed': 1},
    # {'fe_layers': 5, 'fc_layers': 1, 'seed': 4},
    # {'fe_layers': 5, 'fc_layers': 1, 'seed': 5},
    # {'fe_layers': 5, 'fc_layers': 2, 'seed': 1},
    # {'fe_layers': 5, 'fc_layers': 2, 'seed': 4},
    # {'fe_layers': 5, 'fc_layers': 2, 'seed': 5},
    # {'fe_layers': 5, 'fc_layers': 3, 'seed': 1},
    # {'fe_layers': 5, 'fc_layers': 3, 'seed': 4},
    # {'fe_layers': 5, 'fc_layers': 3, 'seed': 5},
    # {'fe_layers': 5, 'fc_layers': 4, 'seed': 1},
    # {'fe_layers': 5, 'fc_layers': 4, 'seed': 4},
    # {'fe_layers': 5, 'fc_layers': 4, 'seed': 5},
    # {'fe_layers': 5, 'fc_layers': 5, 'seed': 1},
    # {'fe_layers': 5, 'fc_layers': 5, 'seed': 4},
    # {'fe_layers': 5, 'fc_layers': 5, 'seed': 5},
    # {'fe_layers': 5, 'fc_layers': 6, 'seed': 1},
    # {'fe_layers': 5, 'fc_layers': 6, 'seed': 4},
    # {'fe_layers': 5, 'fc_layers': 6, 'seed': 5},
    # {'fe_layers': 5, 'fc_layers': 7, 'seed': 1},
    # {'fe_layers': 5, 'fc_layers': 7, 'seed': 4},
    # {'fe_layers': 5, 'fc_layers': 7, 'seed': 5},
    # {'fe_layers': 5, 'fc_layers': 8, 'seed': 1},
    # {'fe_layers': 5, 'fc_layers': 8, 'seed': 4},
    # {'fe_layers': 5, 'fc_layers': 8, 'seed': 5},
    # {'fe_layers': 6, 'fc_layers': 1, 'seed': 1},
    # {'fe_layers': 6, 'fc_layers': 1, 'seed': 4},
    # {'fe_layers': 6, 'fc_layers': 1, 'seed': 5},
    # {'fe_layers': 6, 'fc_layers': 2, 'seed': 1},
    # {'fe_layers': 6, 'fc_layers': 2, 'seed': 4},
    # {'fe_layers': 6, 'fc_layers': 2, 'seed': 5},
    # {'fe_layers': 6, 'fc_layers': 3, 'seed': 1},
    # {'fe_layers': 6, 'fc_layers': 3, 'seed': 4},
    # {'fe_layers': 6, 'fc_layers': 3, 'seed': 5},
    # {'fe_layers': 6, 'fc_layers': 4, 'seed': 1},
    # {'fe_layers': 6, 'fc_layers': 4, 'seed': 4},
    # {'fe_layers': 6, 'fc_layers': 4, 'seed': 5},
    # {'fe_layers': 6, 'fc_layers': 5, 'seed': 1},
    # {'fe_layers': 6, 'fc_layers': 5, 'seed': 4},
    # {'fe_layers': 6, 'fc_layers': 5, 'seed': 5},
    # {'fe_layers': 6, 'fc_layers': 6, 'seed': 1},
    # {'fe_layers': 6, 'fc_layers': 6, 'seed': 4},
    # {'fe_layers': 6, 'fc_layers': 6, 'seed': 5},
    # {'fe_layers': 6, 'fc_layers': 7, 'seed': 1},
    # {'fe_layers': 6, 'fc_layers': 7, 'seed': 4},
    # {'fe_layers': 6, 'fc_layers': 7, 'seed': 5},
    # {'fe_layers': 6, 'fc_layers': 8, 'seed': 1},
    # {'fe_layers': 6, 'fc_layers': 8, 'seed': 2},
    # {'fe_layers': 6, 'fc_layers': 8, 'seed': 4},
    # {'fe_layers': 6, 'fc_layers': 8, 'seed': 5},
    # {'fe_layers': 7, 'fc_layers': 1, 'seed': 1},
    # {'fe_layers': 7, 'fc_layers': 1, 'seed': 4},
    # {'fe_layers': 7, 'fc_layers': 1, 'seed': 5},
    # {'fe_layers': 7, 'fc_layers': 2, 'seed': 1},
    # {'fe_layers': 7, 'fc_layers': 2, 'seed': 4},
    # {'fe_layers': 7, 'fc_layers': 2, 'seed': 5},
    # {'fe_layers': 7, 'fc_layers': 3, 'seed': 1},
    # {'fe_layers': 7, 'fc_layers': 3, 'seed': 4},
    # {'fe_layers': 7, 'fc_layers': 3, 'seed': 5},
    # {'fe_layers': 7, 'fc_layers': 4, 'seed': 1},
    # {'fe_layers': 7, 'fc_layers': 4, 'seed': 4},
    # {'fe_layers': 7, 'fc_layers': 4, 'seed': 5},
    # {'fe_layers': 7, 'fc_layers': 5, 'seed': 1},
    # {'fe_layers': 7, 'fc_layers': 5, 'seed': 4},
    # {'fe_layers': 7, 'fc_layers': 5, 'seed': 5},
    # {'fe_layers': 7, 'fc_layers': 6, 'seed': 4},
    # {'fe_layers': 7, 'fc_layers': 6, 'seed': 5},
    # {'fe_layers': 7, 'fc_layers': 7, 'seed': 4},
    # {'fe_layers': 7, 'fc_layers': 7, 'seed': 5},
    # {'fe_layers': 7, 'fc_layers': 8, 'seed': 4},
    # {'fe_layers': 7, 'fc_layers': 8, 'seed': 5},
    # {'fe_layers': 8, 'fc_layers': 1, 'seed': 4},
    # {'fe_layers': 8, 'fc_layers': 1, 'seed': 5},
    # {'fe_layers': 8, 'fc_layers': 2, 'seed': 4},
    # {'fe_layers': 8, 'fc_layers': 2, 'seed': 5},
    # {'fe_layers': 8, 'fc_layers': 3, 'seed': 4},
    # {'fe_layers': 8, 'fc_layers': 3, 'seed': 5},
    # {'fe_layers': 8, 'fc_layers': 4, 'seed': 4},
    # {'fe_layers': 8, 'fc_layers': 4, 'seed': 5},
    # {'fe_layers': 8, 'fc_layers': 5, 'seed': 4},
    # {'fe_layers': 8, 'fc_layers': 5, 'seed': 5},
    # {'fe_layers': 8, 'fc_layers': 6, 'seed': 4},
    # {'fe_layers': 8, 'fc_layers': 6, 'seed': 5},
    # {'fe_layers': 8, 'fc_layers': 7, 'seed': 4},
    # {'fe_layers': 8, 'fc_layers': 7, 'seed': 5},
    # {'fe_layers': 8, 'fc_layers': 8, 'seed': 4},
    # {'fe_layers': 8, 'fc_layers': 8, 'seed': 5}
    # ]
    # Fig. 6: FE x FC grid (1-8 x 1-8) for Model A (flexicnn) and Model B
    # (linearcnn) over seeds 1-5, as logged by the published runs
    # (result/2026-03-02_17-05-02 and result/2026-03-02_23-29-53).
    inpt_params.setdefault('img_size', 128)
    inpt_params.setdefault('lw', 3)
    inpt_params.setdefault('epochs', 100)
    inpt_params.setdefault('batch_size', 64)
    inpt_params.setdefault('num_workers', 32)
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('learning_rate', 0.001)
    inpt_params.setdefault('fe_layers', [1, 2, 3, 4, 5, 6, 7, 8])
    inpt_params.setdefault('fc_layers', [1, 2, 3, 4, 5, 6, 7, 8])

    inpt_params.setdefault('model', ['flexicnn', 'linearcnn'])

    inpt_params.setdefault('all_measurements', True)
    inpt_params.setdefault('num_rows', None)
    inpt_params.setdefault('data_dir', str(PathsConfig.data_dir))
    inpt_params.setdefault('class_labels', None)
    inpt_params.setdefault('test_aggregation', 'all')
    inpt_params.setdefault('val_aggregation', 'none')
    inpt_params.setdefault('persistent_workers', True)
    inpt_params.setdefault('prefetch_factor', 2)

    keys = list(inpt_params.keys())
    values = [v if isinstance(v, list) else [v] for v in inpt_params.values()]

    log_names: List[str | Path] = []
    for seed in [1, 2, 3, 4, 5]:
        # init dataset with seed to spare the time of reading it multiple times
        params = inpt_params.copy()
        params['seed'] = seed
        params['transformations'] = [
            {"name": "LoopToTensor"},
            {"name": "AWG", "max_mu": 0.15, "sigma": 0.2, "p": 0.75},
            {"name": "SaltAndPepperNoise", "salt_prob": 0.03,
             "pepper_prob": 0.03, "max_amplitude": 1.0, "p": 0.75},
            {"name": "StretchLoops", "max_delta_r": 0.3,
             "max_num_of_axes": 2, "p": 0.75},
            {"name": "ShiftLoops", "max_shift": 0.1,
             "max_num_of_axes": 2, "p": 0.75},
            {"name": "ToImage", "img_size": inpt_params["img_size"], "linewidth": inpt_params["lw"],
             "img_channels": ['f;abs', 'f;rel', 'v;rel'], "num_skips": 3, "max_len_skips": 8, "change_lwidth": False},
            {"name": "RandomRotation", "degree": 8, "p": 0.7},
            {"name": "GaussianBlur", "kernel_size": (
                3, 3), "sigma": (0.1, 1), "p": 0.5}
        ]
        params.setdefault('ttv_ratio', [0.6, 0.2, 0.2])
        set_seed(params["seed"])

        check_param_compatibility(params)

        # train_dataloader, validation_dataloader, test_dataloader, ib_dataset = init_dataset(
        #     params)

        # params["class_labels"] = ib_dataset.class_labels
        for combination in product(*values):

            start_time = time.time()
            # try:
            params = dict(zip(keys, combination))
            params['seed'] = seed
            if params['model'] == 'flexicnn':
                params.setdefault('base_channels', 16)
            elif params['model'] == 'linearcnn':
                # base_channels is not used by linearcnn; 1 is the logged value
                params.setdefault('base_channels', 1)
                params.setdefault('growth_factor', 1.5)
                params.setdefault('bottleneck_channels', 16)

            # Need_to_run = False
            # for conf in missing_configs:
            #     if conf["fe_layers"] == params["fe_layers"] and conf["fc_layers"] == params["fc_layers"] and conf["seed"] == params["seed"]:
            #         cprint(
            #             f"Running with parameters: {params}", "underline_blue")
            #         Need_to_run = True
            #         break
            # if not Need_to_run:
            #     cprint(f"Skipping with parameters: {params}", "yellow")
            #     continue

            cprint(str(params), "underline_blue")

            object = ArchExp(params=params)
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

            object.run(resume=False, train_dataloader=None,
                       validation_dataloader=None, class_labels=None)
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


def run_experiment_arch(inpt_params: dict, resume: bool = False):
    inpt_params.setdefault('img_size', [128, 224])
    inpt_params.setdefault('img_size', 128)
    inpt_params.setdefault('lw', 3)
    inpt_params.setdefault('epochs', 100)
    # inpt_params.setdefault('batch_size', 16)
    # inpt_params.setdefault('num_workers', 4)
    inpt_params.setdefault('batch_size', 256)
    inpt_params.setdefault('num_workers', 6)
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('learning_rate', 0.0005)
    inpt_params.setdefault('model', 'flexicnn')
    # inpt_params.setdefault('model', 'flexicnn')
    # inpt_params.setdefault('model', 'flexicnn')

    # inpt_params.setdefault('fe_layers', [1, 2, 3, 4, 5, 6, 7, 8])
    # inpt_params.setdefault('fc_layers', [1, 2, 3, 4, 5, 6, 7, 8])
    inpt_params.setdefault('fe_layers', [6, 7, 8])
    inpt_params.setdefault('fc_layers', [6, 7, 8])
    # inpt_params.setdefault('base_channels', 1)
    inpt_params.setdefault('base_channels', 16)

    inpt_params.setdefault('seed', [1, 2, 3, 4, 5])
    inpt_params.setdefault('all_measurements', True)
    inpt_params.setdefault('num_rows', None)
    inpt_params.setdefault('data_dir', str(PathsConfig.data_dir))
    inpt_params.setdefault('class_labels', None)
    inpt_params.setdefault('test_aggregation', 'all')
    inpt_params.setdefault('val_aggregation', 'none')
    inpt_params.setdefault('persistent_workers', False)
    inpt_params.setdefault('prefetch_factor', 2)

    keys = list(inpt_params.keys())
    values = [v if isinstance(v, list) else [v] for v in inpt_params.values()]

    log_names: List[str | Path] = []
    for combination in product(*values):
        start_time = time.time()
        # try:
        params = dict(zip(keys, combination))
        if (params["fc_layers"] == 8 and params["fe_layers"] == 1) or (params["fc_layers"] == 1 and params["fe_layers"] == 8):
            continue
        # cprint(str(params), "underline_blue")
        object = ArchExp(params=params)
        # object.run(resume=resume)
        log_names.append(object.log_filename)
        del object
        torch.cuda.empty_cache()
        gc.collect()
        # except Exception as e:
        #     print(f"Error with parameters: {e}")
        end_time = time.time()  # ⏱️ End timer
        elapsed = end_time - start_time
        print(f"Total runtime: {elapsed:.2f} seconds")
    cprint("Experiment completed. Log files:")
    for seed in [1, 2, 3, 4, 5]:
        # init dataset with seed to spare the time of reading it multiple times
        params = inpt_params.copy()
        params['transformations'] = [
            {"name": "LoopToTensor"},
            {"name": "AWG", "max_mu": 0.15, "sigma": 0.2, "p": 0.75},
            {"name": "SaltAndPepperNoise", "salt_prob": 0.03,
             "pepper_prob": 0.03, "max_amplitude": 1.0, "p": 0.75},
            {"name": "StretchLoops", "max_delta_r": 0.3,
             "max_num_of_axes": 2, "p": 0.75},
            {"name": "ShiftLoops", "max_shift": 0.1,
             "max_num_of_axes": 2, "p": 0.75},
            {"name": "ToImage", "img_size": inpt_params["img_size"], "linewidth": inpt_params["lw"],
             "img_channels": ['f;abs', 'f;rel', 'v;rel'], "num_skips": 3, "max_len_skips": 8, "change_lwidth": False},
            {"name": "RandomRotation", "degree": 8, "p": 0.7},
            {"name": "GaussianBlur", "kernel_size": (
                3, 3), "sigma": (0.1, 1), "p": 0.5}
        ]
        params.setdefault('ttv_ratio', [0.6, 0.2, 0.2])
        print(f"Running with seed {params['seed']}...")
        train_dataloader, validation_dataloader, test_dataloader, ib_dataset = init_dataset(
            params)
        params["class_labels"] = ib_dataset.class_labels
        for combination in product(*values):
            start_time = time.time()
            # try:
            params = dict(zip(keys, combination))
            params['seed'] = seed
            # cprint(str(params), "underline_blue")

            object = ArchExp(params=params)
            # object.run(resume=resume, train_dataloader=train_dataloader,
            #            validation_dataloader=validation_dataloader, class_labels=ib_dataset.class_labels)
            log_names.append(object.log_filename)
            del object
            torch.cuda.empty_cache()
            gc.collect()
            # except Exception as e:
            #     print(f"Error with parameters: {e}")
            end_time = time.time()  # ⏱️ End timer
            elapsed = end_time - start_time
            print(f"Total runtime: {elapsed:.2f} seconds")
    cprint("Experiment completed. Log files:")
    for log_name in log_names:
        cprint(str(log_name), "highlighted_green")
    return log_names
