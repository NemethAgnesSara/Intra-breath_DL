import gc
import os
from pathlib import Path
from typing import List

import torch
from IB_DL.experiments.default_config import DefaultParams
from itertools import product
import time
from IB_DL.utils import cprint
from IB_DL.config import PathsConfig, _REPO_ROOT

class EffExp(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        super().__init__()
        params.setdefault('prefix', f"model_" + str(params["model"])+"_img-ize_" + str(
            params["img_size"])+"_seed_" + str(params["seed"])+"_")
        cprint(f"Name: {params['prefix']}", "highlighted_cyan")

        #universal path configuration
        global_store = PathsConfig.output_dir
        current_script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        try:
            relative_path = current_script_dir.relative_to(_REPO_ROOT)
        except ValueError:
            relative_path = Path("custom_experiment")
        base_output = global_store / relative_path / "result"
        if "experiment_date" in params:
            final_output_path = base_output / params["experiment_date"]
        else:
            final_output_path = base_output
        if 'output_dir' not in params:
            params['output_dir'] = str(final_output_path)

        if params is not None:
            self.set_params(params)
        os.makedirs(self.output_dir, exist_ok=True)


def run_experiment(inpt_params: dict, resume: bool = False):
    # Table 2: EfficientNet B0-B3, each at its native resolution. Settings are
    # those logged by the published runs (result/2026-01-26_06-44-15).
    models = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3']
    img_sizes = [224, 240, 288, 320]

    inpt_params.setdefault('epochs', 100)
    inpt_params.setdefault('batch_size', 32)
    inpt_params.setdefault('num_workers', 20)
    inpt_params.setdefault('persistent_workers', True)
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('learning_rate', 0.01)
    # inpt_params.setdefault('seed', [1,2,3,4,5])
    # inpt_params.setdefault('model', ['efficientnet_b0, efficientnet_b1,', 'efficientnet_b2', 'efficientnet_b3',])


    all_params = []
    
    
    for i in range(len(models)):
        param_dict = inpt_params.copy()
        param_dict['model'] = models[i]
        param_dict['img_size'] = img_sizes[i]
        all_params.append(param_dict)



    log_names = []
    for seed in [1, 2, 3, 4, 5]:
        for params in all_params:
            start_time = time.time()
            params['seed'] = seed
            cprint(str(params), "underline_blue")
            object = EffExp(params=params)
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
