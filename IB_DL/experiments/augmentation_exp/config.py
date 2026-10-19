import os
from pathlib import Path
from typing import List, Optional
from matplotlib import pyplot as plt
from IB_DL.config import _REPO_ROOT, PathsConfig
from IB_DL.experiments.default_config import DefaultParams
import numpy as np
from itertools import product
from IB_DL import eval
import time
from IB_DL.utils import cprint
import torch
import gc


class AugmentationExp(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        # It is advised to set prefix in params to show the changed parameters
        super().__init__()
        assert 'transformations' in params, "Transformations must be provided in params"
        self.transformations = params["transformations"]
        # rnd = np.random.randint(0,10000)
        # params['seed']= rnd
        # self.seed =  rnd
        params.setdefault('prefix', f"p_" + str(self.transformations[2]["p"])+"_strength_" + str(
            int(self.transformations[5]["max_len_skips"]/2))+"_rnd_" + str(params["seed"])+"_")
        cprint(f"Name: {params['prefix']}", "highlighted_cyan")
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
        # directory (as each Fig. 8 column had in the published runs).
        final_output_path = final_output_path / params["model"]
        params['output_dir'] = str(final_output_path)

        if params is not None:
            self.set_params(params)
        os.makedirs(self.output_dir, exist_ok=True)
        if 'flexicnn' in self.model:
            self.fe_layers = params.get("fe_layers", 4)
            self.fc_layers = params.get("fc_layers", 4)
            self.base_channels = params.get("base_channels", 16)


# Fig. 8 columns, with the settings logged by the published runs:
# result/2026-01-28_16-01-38 (ResNet-50), result/2026-02-02_10-43-09 (Model A)
# and result/Vendi_run_lin (Model B). All three used lw 3, Adam, lr 0.0005.
ARMS = {
    'resnet50_pretrained': {'img_size': 224, 'batch_size': 128, 'num_workers': 16},
    'flexicnn': {'img_size': 128, 'batch_size': 64, 'num_workers': 24,
                 'fe_layers': 4, 'fc_layers': 4, 'base_channels': 16},
    'linearcnn': {'img_size': 128, 'batch_size': 64, 'num_workers': 32,
                  'fe_layers': 6, 'fc_layers': 5, 'growth_factor': 1.5,
                  'bottleneck_channels': 16},
}


def run_experiment(inpt_params: dict, resume: bool = False):
    inpt_params.setdefault('lw', 3)
    inpt_params.setdefault('epochs', 100)
    inpt_params.setdefault('persistent_workers', True)
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('learning_rate', 0.0005)
    inpt_params.setdefault('seed', [1, 2, 3, 4, 5])
    inpt_params.setdefault('model', list(ARMS))

    models = inpt_params.pop('model')
    models = models if isinstance(models, list) else [models]
    log_names: List[str | Path] = []
    for model in models:
        # explicitly passed parameters take precedence over the arm settings
        arm_params = {**ARMS.get(model, {}), **inpt_params, 'model': model}
        arm_params.setdefault('transformations', _transformations(
            arm_params['img_size'], arm_params['lw']))
        log_names += _run_grid(arm_params, resume)

    for log_name in log_names:
        cprint(str(log_name), "highlighted_green")
    return log_names


def _transformations(img_size: int, lw: int) -> list[list[dict]]:
    """The 5 x 5 grid of execution probability x transformation strength."""
    all_transformations = []
    num_columns = 5
    p = [0.05, 0.25, 0.5, 0.75, 1.0]

    max_mu = np.linspace(0.05, 0.5, num=num_columns)
    sigma = np.linspace(0.1, 1, num=num_columns)
    salt_pepper_prob = np.linspace(0.05, 0.2, num=num_columns)
    max_stretch = np.linspace(0.05, 0.3, num=num_columns)
    max_shift = np.linspace(0.05, 0.3, num=num_columns)
    num_skips = np.array([1, 2, 2, 3, 4])
    max_len_skips = np.linspace(2, 10, num=num_columns)
    random_rot_degree = np.linspace(1, 10, num=num_columns)
    sigma_min_blur = np.linspace(0.05, 0.2, num=num_columns)
    sigma_delta_blur = np.linspace(0.1, 0.5, num=num_columns)
    for p_ in p:
        for i in range(len(max_mu)):
            all_transformations.append([
                {"name": "LoopToTensor"},
                {"name": "AWG", "max_mu": max_mu[i],
                    "sigma": sigma[i], "p": p_},
                {"name": "SaltAndPepperNoise", "salt_prob": salt_pepper_prob[i],
                    "pepper_prob": salt_pepper_prob[i], "max_amplitude": 1.0, "p": p_},
                {"name": "StretchLoops", "max_delta_r": max_stretch[i],
                    "max_num_of_axes": 2, "p": p_},
                {"name": "ShiftLoops", "max_shift": max_shift[i],
                    "max_num_of_axes": 2, "p": p_},
                {"name": "ToImage", "img_size": img_size, "linewidth": lw,
                    "img_channels": ['f;abs', 'f;rel', 'v;rel'], "num_skips": num_skips[i], "max_len_skips": max_len_skips[i], "change_lwidth": False},
                {"name": "RandomRotation",
                    "degree": random_rot_degree[i], "p": p_},
                {"name": "GaussianBlur", "kernel_size": (
                    3, 3), "sigma": (sigma_min_blur[i], sigma_min_blur[i]+sigma_delta_blur[i]), "p": p_}
            ])
    return all_transformations


def _run_grid(inpt_params: dict, resume: bool) -> List[str | Path]:
    keys = list(inpt_params.keys())
    values = [v if isinstance(v, list) else [v] for v in inpt_params.values()]

    log_names: List[str | Path] = []
    for combination in product(*values):
        start_time = time.time()
        params = dict(zip(keys, combination))
        object = AugmentationExp(params=params)
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
    return log_names


if __name__ == "__main__":
    print("Default experiment")
    log_names: list[str | Path] = ["/home/sacz/projects/IB_DL/IB_DL/experiments/default_exp/result/epochs_50__model_resnet18__batch_size_4__learning_rate_0.01__log__2025-08-01_11-50-59.log",
                                   "/home/sacz/projects/IB_DL/IB_DL/experiments/default_exp/result/epochs_50__model_resnet18__batch_size_128__learning_rate_0.01__log__2025-08-01_12-31-04.log"]
    eval.evaluate(log_files=log_names)


# class AugExp(DefaultParams):
#     def __init__(self, params: dict):
#         """
#         Initialize the experiment with parameters.
#         :param params: Dictionary of parameters to override the default ones.
#         """
#         # It is advised to set prefix in params to show the changed parameters
#         super().__init__()
#         params.setdefault('prefix', "_".join(
#             f"{k}_{str(v).replace(' ', '_')}_" for k, v in params.items()))
#         cprint(f"Name: {params['prefix']}", "highlighted_cyan")
#         if "experiment_date" in params.keys():
#             params.setdefault('output_dir', str(
#                 Path(os.path.dirname(os.path.realpath(__file__))) / "result" / params["experiment_date"]))
#         else:
#             params.setdefault('output_dir', str(
#                 Path(os.path.dirname(os.path.realpath(__file__))) / "result"))
#         if params is not None:
#             self.set_params(params)

#         self.transformations: list = [
#             {"name": "LoopToTensor"},
#             {"name": "AWG", "max_mu": 0.0, "sigma": 0.1, "p": 0.35}, #nem több idő --model "baseline" --epochs 5 --img-size 64 11.94 sec
#             {"name": "SaltAndPepperNoise", "salt_prob": 0.01,
#                 "pepper_prob": 0.01, "max_amplitude": 1.0, "p": 0.35},#nem több idő
#              {"name": "StretchLoops", "max_delta_r": 0.2,
#                 "max_num_of_axes": 2, "p": 0.5}, ##11-93
#              {"name": "ShiftLoops", "max_shift": 0.2,
#                 "max_num_of_axes": 2, "p": 0.5}, ## 12.11sec
#             {"name": "ToImage", "img_size": self.img_size, "linewidth": self.lw,
#                 "img_channels":  ['f;abs', 'f;rel', 'f;rel'], "num_skips": 1, "max_len_skips": 5},
#             {"name": "RandomRotation", "degree": 7.5, "p": 0.5}, ##20
#             {"name": "GaussianBlur", "kernel_size" : (5,7), "sigma": (0.1,1), "p": 0.5}]
#         self.epochs = 1
#         os.makedirs(self.output_dir, exist_ok=True)


# def run_experiment(inpt_params: dict):
#     keys = list(inpt_params.keys())
#     values = [v if isinstance(v, list) else [v] for v in inpt_params.values()]

#     log_names: List[str | Path] = []
#     for combination in product(*values):
#         start_time = time.time()
#         try:
#             params = dict(zip(keys, combination))
#             cprint(str(params), "underline_blue")
#             object = AugExp(params=params)
#             object.run()
#             log_names.append(object.log_filename)
#         except Exception as e:
#             print(f"Error with parameters: {e}")
#         end_time = time.time()  # ⏱️ End timer
#         elapsed = end_time - start_time
#         print(f"Total runtime: {elapsed:.2f} seconds")
#     cprint("Experiment completed. Log files:")
#     for log_name in log_names:
#         cprint(str(log_name), "highlighted_green")
#     return log_names


# if __name__ == "__main__":
#     print("Default experiment")
#     # log_names: list[str | Path] = ["/home/sacz/projects/IB_DL/IB_DL/experiments/default_exp/result/epochs_50__model_resnet18__batch_size_4__learning_rate_0.01__log__2025-08-01_11-50-59.log",
#     #                                "/home/sacz/projects/IB_DL/IB_DL/experiments/default_exp/result/epochs_50__model_resnet18__batch_size_128__learning_rate_0.01__log__2025-08-01_12-31-04.log"]
#     # eval.evaluate(log_files=log_names)
#     run_experiment({})
