import gc
import os
from pathlib import Path
from typing import List, Optional
from IB_DL.config import _REPO_ROOT, PathsConfig
from IB_DL.experiments.default_config import DefaultParams
import torch
from itertools import product
import time
from IB_DL.utils import cprint


class PreTrain(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        # It is advised to set prefix in params to show the changed parameters
        super().__init__()

        params.setdefault('prefix', f"model_" +
                          str(params["model"])+"_rnd_" + str(params["seed"])+"_")
        cprint(f"Name: {params['prefix']}", "highlighted_cyan")
       # universal path configuration
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
        


# Line width of each Table 3 row, as logged by the published runs:
# result/2026-01-27_14-33-50 (Stylized-ImageNet and Quick, Draw! pretraining)
# and default_exp/result/2026-01-23_15-03-16 (no intermediate training).
# Augmentation is the default tf_strength 2 pipeline.
LINE_WIDTH = {
    'resnet50_SIN': 3, 'resnet50_SIN_IMGNET': 3,
    'QD_resnet50': 3, 'QD_pretrained_resnet50': 3,
    'resnet50': 2, 'resnet50_pretrained': 2,
}


def run_experiment(inpt_params: dict, resume: bool = False):
    lw_given = 'lw' in inpt_params
    inpt_params.setdefault('img_size', 224)
    inpt_params.setdefault('lw', 3)
    inpt_params.setdefault('epochs', 100)
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('learning_rate', 0.001)
    inpt_params.setdefault('model', list(LINE_WIDTH))
    inpt_params.setdefault('seed', [1, 2, 3, 4, 5])
    inpt_params.setdefault('batch_size', 128)
    inpt_params.setdefault('num_workers', 16)
    inpt_params.setdefault('persistent_workers', True)

    keys = list(inpt_params.keys())
    values = [v if isinstance(v, list) else [v] for v in inpt_params.values()]

    log_names: List[str | Path] = []
    for combination in product(*values):
        start_time = time.time()
        params = dict(zip(keys, combination))
        if not lw_given:
            params['lw'] = LINE_WIDTH.get(params['model'], params['lw'])
        cprint(str(params), "underline_blue")
        object = PreTrain(params=params)
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


if __name__ == "__main__":
    run_experiment({})
