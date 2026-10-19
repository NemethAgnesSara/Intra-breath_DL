"""Does the model already read breath direction out of the loop geometry?

The temporal-encoding A/B found no reliable accuracy gain. One explanation is
that the ramp adds nothing the network could not already infer, because the
channel stack carries the direction cues implicitly. This experiment tests that
by removing the cues one at a time and asking whether the gradient starts to
matter once they are gone.

Two cues let a static loop be read directionally:

  * a *within-phase clock* - volume is monotonic inside each phase, so position
    along the volume axis says how far through the phase the breath is; and
  * a *phase label* - the sign of flow says whether this is inspiration or
    expiration, but flow is not monotonic within a phase.

    Arm A  ['v;abs', 'v;rel']   volume only - clock, no phase label
    Arm B  ['f;abs', 'f;rel']   flow only   - phase label, no clock
    Arm C  ['f;abs', 'v;abs']   one of each - both, direction recoverable

Prediction: temporal encoding should help in A and B, where a cue is missing,
and should not help in C, where the geometry already determines direction.

Design: gradient {off, on} x arm {A, B, C} x model {flexicnn, linearcnn}
x 5 seeds = 60 runs. Everything else matches the temporal-encoding A/B in
``experiments/gradient``: Adam, mild augmentation (tf_strength 1), 128px / lw 2.

Lives in its own experiment directory rather than under ``gradient`` because the
CLI resolves ``experiments/<name>/config.py``, and because mixing these runs into
that result folder would break ``gradient/analysis.py``, which pairs runs on
(model, seed, aggregation) and assumes a single channel configuration.
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

# The direction cues each arm carries; see the module docstring.
ARM_A_VOLUME_ONLY = ['v;abs', 'v;rel']
ARM_B_FLOW_ONLY = ['f;abs', 'f;rel']
ARM_C_BOTH = ['f;abs', 'v;abs']


class ChannelExp(DefaultParams):
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
    one of the two things this experiment varies and must never fall back to a
    default.
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

    # --- the two variables --------------------------------------------------
    inpt_params.setdefault('gradient', [False, True])
    inpt_params.setdefault('img_channels', [ARM_A_VOLUME_ONLY,
                                            ARM_B_FLOW_ONLY,
                                            ARM_C_BOTH])

    # --- held fixed, matching experiments/gradient --------------------------
    inpt_params.setdefault('model', ['flexicnn', 'linearcnn'])
    inpt_params.setdefault('optimizer', "Adam")
    inpt_params.setdefault('tf_strength', 1)
    inpt_params.setdefault('img_size', 128)
    inpt_params.setdefault('lw', 2)
    inpt_params.setdefault('base_channels', 16)
    inpt_params.setdefault('seed', [1, 2, 3, 4, 5])

    # --- training setup ------------------------------------------------------
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
        if params['lw'] < 1:
            params['lw'] = 1

        # Built here, not before the loop: the block above has just set the
        # per-model img_size/lw, and the pipeline has to use those.
        gradient = params.pop('gradient')
        params['transformations'] = _transformations(
            params['img_size'], params['lw'], params['img_channels'], gradient)

        # Both varied dimensions go in the prefix: the "already exists" guard
        # below is prefix-keyed, model_label in test_results.csv is the prefix,
        # and the arms would otherwise overwrite each other's checkpoints.
        params['prefix'] = (
            "model_" + str(params["model"]) + "_optimizer_" +
            str(params["optimizer"]) + "_tf-trength_" + str(params["tf_strength"]) +
            "_img-channels_" + str(params["img_channels"]) +
            "_gradient_" + str(gradient) + "_seed_" + str(params["seed"]) + "_")

        cprint(str(params), "underline_blue")
        object = ChannelExp(params=params)
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
