from datetime import date, datetime
from genericpath import exists
from pathlib import Path
from typing import Optional
import torch
from IB_DL.config import PathsConfig
import os
from IB_DL.train import train
from IB_DL.utils import cprint


class DefaultParams:
    def __init__(self):
        self.model: str = 'baseline'
        self.optimizer: str = 'SGD'
        self.epochs: int = 10
        self.batch_size: int = 32
        self.learning_rate: float = 0.001
        self.data_dir: str | Path = str(PathsConfig.data_dir)
        self.output_dir: str | Path = str(os.path.dirname(
            os.path.abspath(__file__))) + "/result"
        self.log_level: str = 'INFO'
        self.seed: int = 42
        self.gpu: bool = torch.cuda.is_available()
        self.resume: bool = False
        self.verbose: bool = True
        self.eval_interval: int = 1
        self.early_stop: bool = True
        self.class_labels: Optional[str] = None
        self.all_measurements: bool = True
        self.ttv_ratio: list[float] = [0.6, 0.2, 0.2]
        self.output_items: list[str] = ['loss_plot',
                                        'acc_plot',
                                        'conf_matrix',
                                        'cams']
        self.prefix: str = ''
        self.suffix: str = f"_{date.today().isoformat()}_{datetime.now().strftime('%H-%M-%S')}"
        self.num_rows: Optional[int] = None
        self.img_size: int = 224
        self.lw: int = 2
        self.img_channels: list[str] = ['f;abs', 'f;rel', 'v;rel']
        self.tf_strength = 2
        self.num_workers = 16
        self.persistent_workers = False
        self.prefetch_factor = 2
        self.test_aggregation: str = 'all'  # 'none' | 'median_loop' | 'mean' | 'all'
        self.val_aggregation: str = 'none'  # 'none' | 'median_loop' | 'mean'
        # 'MEDIAN LOOP' | 'ARTIFICIAL MEDIAN LOOP' | 'MEAN' | 'MEDIAN' | None
        self.feature_selector: Optional[str] = None
        self.early_counter = 10
        self.warmup_epochs = 3
        self.fe_layers: int = 8
        self.fc_layers: int = 4
        self.base_channels: int = 8
        self.min_epochs: int = 15

        self.start_epoch: int = 0
        # Initialize with None, needed in case of resuming
        self.early_last_state: Optional[int] = None
        self.best_val_accuracy: Optional[float] = None
        self.best_val_loss: Optional[float] = None
        self.lr_scheduler_last_state: Optional[int] = None

    def set_filenames(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        if not hasattr(self, 'log_filename') or self.log_filename is None:
            self.log_filename: str | Path = os.path.join(
                self.output_dir, f"{self.prefix}_log_{self.suffix}.log")
        if not hasattr(self, 'best_model_filename') or self.best_model_filename is None:
            self.best_model_filename: str | Path = os.path.join(
                self.output_dir, f"{self.prefix}_log_{self.suffix}.pt")

    def to_dict(self, set_filenames: Optional[bool] = True):
        if set_filenames:
            self.set_filenames()
        return {k: v for k, v in self.__dict__.items()}

    def run(
            self, resume: bool = False, train_dataloader=None, validation_dataloader=None,
            class_labels=None, example_img=False):
        train(self.to_dict(), resume=resume, train_dataloader=train_dataloader,
              validation_dataloader=validation_dataloader, class_labels=class_labels)

    def refreash_params(self):

        if not hasattr(
                self, 'transformations') or not isinstance(
                self.transformations, list):
            if self.tf_strength == 2:
                self.transformations: list[dict] = [
                    {"name": "LoopToTensor"},
                    {"name": "AWG", "max_mu": 0.15, "sigma": 0.2, "p": 0.75},
                    {"name": "SaltAndPepperNoise", "salt_prob": 0.03,
                        "pepper_prob": 0.03, "max_amplitude": 1.0, "p": 0.75},
                    {"name": "StretchLoops", "max_delta_r": 0.3,
                        "max_num_of_axes": 2, "p": 0.75},
                    {"name": "ShiftLoops", "max_shift": 0.1,
                        "max_num_of_axes": 2, "p": 0.75},
                    {"name": "ToImage", "img_size": self.img_size, "linewidth": self.lw,
                        "img_channels": self.img_channels, "num_skips": 3, "max_len_skips": 8, "change_lwidth": False},
                    {"name": "RandomRotation", "degree": 8, "p": 0.7},
                    {"name": "GaussianBlur", "kernel_size": (
                        3, 3), "sigma": (0.1, 1), "p": 0.5}
                ]
            elif self.tf_strength == 1:
                self.transformations: list[dict] = [
                    {"name": "LoopToTensor"},
                    {"name": "AWG", "max_mu": 0.1, "sigma": 0.1, "p": 0.25},
                    {"name": "SaltAndPepperNoise", "salt_prob": 0.01,
                        "pepper_prob": 0.01, "max_amplitude": 1.0, "p": 0.25},
                    {"name": "StretchLoops", "max_delta_r": 0.15,
                        "max_num_of_axes": 2, "p": 0.25},
                    {"name": "ShiftLoops", "max_shift": 0.05,
                        "max_num_of_axes": 2, "p": 0.25},
                    {"name": "ToImage", "img_size": self.img_size, "linewidth": self.lw,
                        "img_channels": self.img_channels, "num_skips": 2, "max_len_skips": 4, "change_lwidth": False},
                    {"name": "RandomRotation", "degree": 5, "p": 0.2},
                    {"name": "GaussianBlur", "kernel_size": (
                        3, 3), "sigma": (0.05, 0.1), "p": 0.2}
                ]

    def set_params(self, params: Optional[dict] = None):
        """
        Set parameters from a dictionary.
        If a parameter is not present in the dictionary, it will not be changed."""
        if params is not None:
            for key, value in params.items():
                if not hasattr(self, key):
                    cprint(f"Warning: {key} is not a valid parameter.", "warning")
                setattr(self, key, value)

        self.set_filenames()
        self.refreash_params()
        self.check_params()

    def check_params(self):
        """Check if the parameters are valid."""
        if not self.all_measurements:
            cprint("Warning: Not all measurements are used.", "warning")
        if self.num_rows is not None:
            cprint(
                f"Number of loops used is set to {self.num_rows}.", "warning")
            self.num_rows = int(self.num_rows)
