import gc
import os
from pathlib import Path
from typing import List, Optional
from unittest import result
from matplotlib import pyplot as plt
from IB_DL.experiments.default_config import DefaultParams
from IB_DL.eval import evaluate
from IB_DL.logger import LogReader
from datetime import date, datetime
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from itertools import product
import time
from IB_DL.utils import cprint


class OneDim(DefaultParams):
    def __init__(self, params: dict):
        """
        Initialize the experiment with parameters.
        :param params: Dictionary of parameters to override the default ones.
        """
        super().__init__()
        assert 'transformations' in params, "Transformations must be provided in params"
        self.transformations = params["transformations"]
        params.setdefault(
            'prefix', f"label"+str(params["label"]) + "_model_" + str(params["model"]) +
            "_seed_" + str(params["seed"]) + "_")
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


def run_experiment(inpt_params: dict, resume: bool = False):
    # # raw data like old times, patient wise extra outlier resistion
    # self.transformations: list[dict] = [ {"name": "ToMeasurements"},
    #                         {"name": "LoopToIBFeatures", "feature_selector": self.feature_selector}]

    # # raw data loop wise
    # self.transformations: list[dict] = [
    #     {"name": "LoopToIBFeatures", "feature_selector": "MEAN", "measurement": False}]

    # raw data 4x50 tesnor wise for 1d conv
    # self.transformations: list[dict] = [ {"name": "LoopToTensor"}]
    # self.transformations: list[dict] = [ {"name": "LoopToTensor"},
    #                                     {"name": "AWG", "max_mu": 0.15, "sigma": 0.4, "p": 0.8},
    #     {"name": "SaltAndPepperNoise", "salt_prob": 0.03,
    #         "pepper_prob": 0.03, "max_amplitude": 1.0, "p": 0.8},
    #     {"name": "StretchLoops", "max_delta_r": 0.3,
    #         "max_num_of_axes": 2, "p": 0.8},
    #     {"name": "ShiftLoops", "max_shift": 0.2,
    #         "max_num_of_axes": 2, "p": 0.8}]
    log_names = []
    all_transformations = [
        # 1 patient-wise, artificial median loop, outlier rejection ON
        [{"name": "ToMeasurements"},
         {"name": "LoopToIBFeatures", "feature_selector": "ARTIFICIAL MEDIAN LOOP",
          "measurement": True}],
        # 2 same, outlier rejection OFF. Note this is switched on ToMeasurements:
        # "measurement": False would NOT do it -- that flag only declares the input
        # type and is forced back to True, which used to make this config a silent
        # duplicate of the one above.
        [{"name": "ToMeasurements", "outlier_detection": False},
         {"name": "LoopToIBFeatures", "feature_selector": "ARTIFICIAL MEDIAN LOOP",
          "measurement": True}],
        # 3 patient-wise, mean of per-loop features, outlier rejection ON
        [{"name": "ToMeasurements"},
         {"name": "LoopToIBFeatures", "feature_selector": "MEAN",
          "measurement": True}],
        # 4 loop-wise feature extraction (no ToMeasurements -> loop_set=True)
        [{"name": "LoopToIBFeatures", "feature_selector": "MEAN",
          "measurement": False}],
        # 5 raw 4x50 tensor
        [{"name": "LoopToTensor"}],
        # 6 raw tensor + augmentation
        [{"name": "LoopToTensor"},
         {"name": "AWG", "max_mu": 0.15, "sigma": 0.4, "p": 0.8},
         {"name": "SaltAndPepperNoise", "salt_prob": 0.03,
          "pepper_prob": 0.03, "max_amplitude": 1.0, "p": 0.8},
         {"name": "StretchLoops", "max_delta_r": 0.3,
          "max_num_of_axes": 2, "p": 0.8},
         {"name": "ShiftLoops", "max_shift": 0.2,
          "max_num_of_axes": 2, "p": 0.8}],
    ]
    labels = ['featuresArtmedNoOutlier', 'featuresArtmedOutlier',
              'featuresMeanNoOutlier', 'featuresLoopwiseMean', 'raw', 'rawAug']
    models = ["fc_nn", "fc_nn", "fc_nn", "fc_nn", "time_cnn", "time_cnn"]
    seeds = [1, 2, 3, 4, 5]

    # all_transformations = [
    #     [{"name": "ToMeasurements"}, {"name": "LoopToIBFeatures",
    #                                   "feature_selector": "MEAN", "measurement": True}],
    #     [{"name": "LoopToIBFeatures", "feature_selector": "MEAN", "measurement": False}],
    #     [{"name": "LoopToTensor"}],
    #     [{"name": "LoopToTensor"}, {"name": "AWG", "max_mu": 0.15, "sigma": 0.4, "p": 0.8},
    #      {"name": "SaltAndPepperNoise", "salt_prob": 0.03,
    #         "pepper_prob": 0.03, "max_amplitude": 1.0, "p": 0.8},
    #         {"name": "StretchLoops", "max_delta_r": 0.3,
    #          "max_num_of_axes": 2, "p": 0.8},
    #         {"name": "ShiftLoops", "max_shift": 0.2,
    #          "max_num_of_axes": 2, "p": 0.8}]

    # ]
    # labels = [
    #           'featuresMeanNoOutlier', 'featuresLoopwiseMean', 'raw', 'rawAug']
    # models = [ "time_cnn", "time_cnn", "fc_nn", "fc_nn"]
    # seeds = [1,2,3,4,5]
    for i in range(len(all_transformations)):
        for j in seeds:
            params = {"epochs": 100,
                      "model": models[i],
                      "optimizer": "Adam",
                      "learning_rate": 0.001,
                      "batch_size": 128,
                      "seed": j,
                      "experiment_date": inpt_params["experiment_date"],
                      "feature_selector": "MEDIAN",
                      "transformations": all_transformations[i],
                      "label": labels[i],
                      "save_patient_split": True}
            object = OneDim(params=params)
            # Skip runs already present in the output dir so the grid is restartable.
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

    return log_names


if __name__ == "__main__":
    experimen_date = date.today().strftime("%Y-%m-%d")
    inpt_params = {"experiment_date": experimen_date}
    log_names = run_experiment(inpt_params=inpt_params, resume=False)
    # Produces test_results.csv with the loop_wise / majority_vote (soft vote) /
    # median_loop accuracies that Table 6 is built from.
    evaluate(log_names)
