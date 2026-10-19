"""
QuickDraw Dataset Integration & Training Module

Data source and handling adapted from: 
https://github.com/googlecreativelab/quickdraw-dataset?tab=readme-ov-file#the-raw-moderated-dataset

Simplified Drawing files format:
- Vectors are simplified, timing information removed.
- Positioned and scaled into a 256x256 region.
- Exported in ndjson format.
"""

from datetime import date, datetime
import numpy as np
import torch
from torchvision.utils import draw_keypoints
import json
import os
import matplotlib.pyplot as plt
from IB_DL.config import PathsConfig, _REPO_ROOT
from torch.utils.data import DataLoader, Dataset
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Optional, Sequence, Union  # type: ignore
from typing import Any, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader  # type: ignore
from IB_DL.datasets import Loop, init_dataset
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from IB_DL.eval import check_accuracy, check_accuracy_no_eval, init_logreaders
from IB_DL.models import get_model
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from IB_DL.logger import Logger
from IB_DL.utils import cprint, set_seed, check_param_compatibility
from IB_DL import utils
from IB_DL.utils import get_optimizer
from torchvision.transforms import v2
from IB_DL.train import train_loop
import pandas as pd
import urllib.request
from urllib.parse import quote

DEFAULT_IMG_SIZE = 224
DEFAULT_CHANNELS = 3
LINE_WIDTH = 3  # Width of the drawn line/stroke

QD_CLASSES = ["apple", "boomerang", "carrot", "cookie", "hot air balloon",
              "mushroom", "onion", "pear", "sun", "wheel"]
QD_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
QD_FILE_BYTES = 20 * 1024 * 1024


def download_quickdraw(data_dir: Optional[str] = None) -> None:
    """
    Downloads the Quick, Draw! classes used for pretraining into <data_dir>/QuickDraw.

    The pretraining in the paper used about the first 20 MB of each class's
    simplified .ndjson file, so each download stops at the end of the line that
    crosses 20 MiB. The files used for the published checkpoints came from an
    earlier download of the same dataset; they share ~97% of their drawings with
    these files but are not byte-identical.
    """
    data_dir = data_dir or str(PathsConfig.data_dir)
    out_dir = os.path.join(data_dir, "QuickDraw")
    os.makedirs(out_dir, exist_ok=True)
    for cls in QD_CLASSES:
        out_file = os.path.join(out_dir, f"{cls}.ndjson")
        if os.path.exists(out_file):
            continue
        written = 0
        with urllib.request.urlopen(QD_URL.format(quote(cls)), timeout=60) as response, \
                open(out_file, "wb") as f:
            for line in response:
                f.write(line)
                written += len(line)
                if written >= QD_FILE_BYTES:
                    break
        print(f"Downloaded {cls}: {written} bytes")


def parse_line(ndjson_line: str) -> tuple[np.ndarray, str]:
    """
    Adapted from github.
    Parses a single line from a QuickDraw .ndjson file.

    Extracts the drawing and formats it into a NumPy array of shape (N, 3), 
    where N is the total number of points across all strokes.

    Features:
    1. delta_x: difference in x coordinate from the previous point.
    2. delta_y: difference in y coordinate from the previous point.
    3. stroke_end_flag: 1 if the point is the end of a stroke, else 0.
    """
    sample = json.loads(ndjson_line)
    class_name = sample["word"]
    inkarray = sample["drawing"]
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)

    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0

    for stroke in inkarray:
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1  # Mark the end of the stroke

    # Preprocessing: Size normalization to 0-1 range
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale

    # Compute deltas
    np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
    np_ink = np_ink[1:, :]

    return np_ink, class_name


def render_ink_to_canvas(
        ink_np_array: np.ndarray, img_size: int = DEFAULT_IMG_SIZE,
        channels: int = DEFAULT_CHANNELS, line_width: int = LINE_WIDTH) -> torch.Tensor:
    """
    Renders the delta-encoded ink array onto a PyTorch tensor canvas. To be compatible with the original loop data, and training pipeline.
    Uses torchvision.draw_keypoints to natively draw the lines on the GPU/CPU tensor.
    """
    if ink_np_array.shape[0] == 0:
        print("Warning: Input ink array is empty. Returning blank canvas.")
        return torch.zeros((channels, img_size, img_size), dtype=torch.float32)

    # Inverse Transformation (Delta to Absolute Coordinates)
    delta_coords = ink_np_array[:, 0:2]
    abs_coords_normalized = np.cumsum(delta_coords, axis=0)

    # Denormalization, Scaling, and Centering
    min_coords = np.min(abs_coords_normalized, axis=0)
    max_coords = np.max(abs_coords_normalized, axis=0)
    drawing_extent = max_coords - min_coords

    PADDING_FACTOR = 0.95
    max_canvas_dim = (img_size - 1) * PADDING_FACTOR

    scale_x = max_canvas_dim / drawing_extent[0]
    scale_y = max_canvas_dim / drawing_extent[1]
    uniform_scale = np.min([scale_x, scale_y])

    scaled_coords = abs_coords_normalized * uniform_scale
    scaled_min = np.min(scaled_coords, axis=0)
    scaled_max = np.max(scaled_coords, axis=0)
    scaled_extent = scaled_max - scaled_min

    center_offset = ((img_size - 1) - scaled_extent) / 2
    abs_coords_pixels = np.round(
        scaled_coords + (center_offset - scaled_min)).astype(np.int64)
    abs_coords_pixels = np.clip(abs_coords_pixels, 0, img_size - 1)

    # Calculate Connectivity for Strokes
    connectivity = []
    N = len(ink_np_array)
    for i in range(N - 1):
        if ink_np_array[i, 2] == 0:
            connectivity.append((i, i + 1))

    # Prepare Canvas and Keypoints Tensor
    initial_canvas_f32 = torch.zeros(
        (channels, img_size, img_size),
        dtype=torch.float32)
    temp_image = (initial_canvas_f32 * 255).to(torch.uint8)
    keypoints_tensor = torch.from_numpy(abs_coords_pixels).unsqueeze(0)

    # Draw the Keypoints
    drawn_image_uint8 = draw_keypoints(
        image=temp_image,
        keypoints=keypoints_tensor,
        connectivity=connectivity,
        colors="white",
        radius=0,
        width=line_width
    )

    final_canvas = drawn_image_uint8.to(torch.float32) / 255.0
    return final_canvas


class QDDataset(Dataset):
    """
    PyTorch Dataset wrapper for the QuickDraw .ndjson files. Compatible with the existing training pipeline, returning (data, label) pairs.
    Handles loading, parsing, and optional transformations of the raw QuickDraw data.
    """

    def __init__(self,
                 csv_file: str | list[str] = "ALL",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:

        if csv_file == "ALL":
            cls_name = ["sun", "wheel", "pear", "onion", "mushroom", "hot air balloon",
                        "cookie", "apple", "carrot", "boomerang"]
            data_dir = str(PathsConfig.data_dir)
            csv_file = [
                os.path.join(data_dir, "QuickDraw", f"{cls}.ndjson") for cls in
                cls_name]

        if not isinstance(csv_file, list):
            csv_file = [csv_file]

        data = []
        labels = []

        for in_file in csv_file:
            if not os.path.exists(in_file):
                raise FileNotFoundError(f"File {in_file} not found.")
            with open(in_file, 'r', encoding='utf-8') as infile:
                for idx, line in enumerate(infile):
                    # Keep only one third of the lines to manage dataset size
                    if idx % 3 != 0:
                        continue
                    obj_line = line.strip()
                    ink, label = parse_line(obj_line)
                    data.append(ink)
                    labels.append(label)

        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

        self.check_data()
        self.class_labels = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_labels)}

    def check_data(self) -> None:
        """Removes empty drawings from the dataset."""
        kept_data = []
        kept_labels = []
        removed_count = 0

        for idx, ink in enumerate(self.data):
            empty = False
            try:
                if isinstance(ink, np.ndarray):
                    empty = ink.size == 0 or (ink.ndim > 0 and ink.shape[0] == 0)
                    if ink.shape[0] < 3:
                        empty = True
                else:
                    empty = len(ink) == 0
            except Exception:
                empty = ink is None

            if empty:
                removed_count += 1
            else:
                kept_data.append(ink)
                kept_labels.append(self.labels[idx])

        if removed_count:
            self.data = kept_data
            self.labels = kept_labels
            print(f"Removed {removed_count} empty drawing(s).")

    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx) -> tuple[Any, int]:
        loops = self.data[idx]
        label = self.labels[idx]
        self.extra_params: Dict[str, Any] = {}

        if self.transform:
            transformed = self.transform(self.data[idx], self.extra_params)
            loops = transformed[0] if isinstance(
                transformed, (list, tuple)) else transformed

            if isinstance(loops, np.ndarray):
                loops = torch.from_numpy(loops)

            if isinstance(loops, torch.Tensor):
                if loops.ndim == 2:
                    loops = loops.unsqueeze(0)
                elif loops.ndim == 3 and loops.shape[0] not in (1, 3) and loops.shape[2] in (1, 3):
                    loops = loops.permute(2, 0, 1)
                loops = loops.to(dtype=torch.float32)

        if self.target_transform:
            label_idx = self.target_transform(label, self.extra_params)
            return loops, label_idx

        return loops, label


class LinesToImage(v2.Transform):
    """
    Torchvision v2 Transform wrapper for render_ink_to_canvas.
    """

    def __init__(
            self, img_size: int = DEFAULT_IMG_SIZE, channels: int = DEFAULT_CHANNELS,
            line_width: int = LINE_WIDTH) -> None:
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        self.line_width = line_width

    def __call__(self, inpt: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        canvas = render_ink_to_canvas(
            inpt, img_size=self.img_size, channels=self.channels,
            line_width=self.line_width)
        my_canvas = torch.zeros(
            (self.channels, self.img_size, self.img_size),
            dtype=torch.float32)
        my_canvas[:, :, :] = canvas[:, :, :]
        my_canvas = torch.flip(my_canvas, dims=[1])
        return my_canvas


def pretrain_qd(params: Dict, resume: bool = False):
    """
    Pre-training orchestration function for the QuickDraw dataset.
    Sets up loaders, models, and hooks into custom train_loop, that is very similar to the one used for the Loop dataset, but adapted for the QuickDraw data format and training requirements.
    """
    set_seed(params["seed"])
    print("Training model with the following parameters:")
    for k, v in params.items():
        cprint(f"  {k}: {v}", "blue")

    qd_dataset = QDDataset(csv_file="ALL", transform=v2.Compose([LinesToImage()]))
    qd_dataset.class_labels = sorted(list(set(qd_dataset.labels)))

    train_frac = float(params.get("train_split", 0.6))
    val_frac = float(params.get("val_split", 0.2))
    total = len(qd_dataset)
    train_size = int(total * train_frac)
    val_size = int(total * val_frac)
    test_size = total - train_size - val_size

    gen = torch.Generator().manual_seed(int(params.get("seed", 0)))
    train_dataset, val_dataset, test_dataset = random_split(
        qd_dataset, [train_size, val_size, test_size], generator=gen
    )

    batch_size = int(params.get("batch_size", 32))
    num_workers = int(params.get("num_workers", 4))
    pin_memory = bool(params.get("gpu", True))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory)
    validation_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory)

    params["class_labels"] = qd_dataset.class_labels
    params["num_classes"] = len(params["class_labels"])
    params["device"] = "cuda" if params["gpu"] else "cpu"

    nn_model = get_model(params)
    if resume:
        cprint("Resuming training from checkpoint.", "red")
        nn_model.load_state_dict(torch.load(
            params["model"], map_location=params["device"]))

    nn_model = nn_model.to(params["device"])
    optimizer = get_optimizer(params["optimizer"], nn_model, params["learning_rate"])
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=params["learning_rate"],
        total_iters=params.get("warmup_epochs", 0))
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, cooldown=2, threshold=0.01
    )

    utils.header = True
    log_file = None
    try:
        log_file = Logger(params["log_filename"])
        log_file.open()
        log_file.write_params(params=params)
        train_loop(
            model=nn_model,
            dataloader=train_dataloader,
            params=params,
            optimizer=optimizer,
            lr_schedulers=(scheduler1, scheduler2),
            val_loader=validation_dataloader,
            log_file=log_file
        )
    except (Exception, KeyboardInterrupt) as e:
        cprint(f"Training stopped: {e}", "error")
        cprint("Saving model state before exiting...", "error")
        model_save_path = os.path.join(
            params["output_dir"],
            f"{params['prefix']}_{params['model']}_model_backup_{params['suffix']}.pt")
        torch.save(nn_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        if log_file is not None:
            log_file.write_event("exception", {"type": str(type(e)), "message": str(e)})
            log_file.write_event("saving", {"epoch": -1})
    finally:
        if log_file is not None:
            log_file.close()
        torch.cuda.empty_cache()


def eval_qd_model(log_file: str) -> None:
    """
    Evaluation function for a trained QuickDraw model. Loads the model checkpoint, prepares the test dataloader, and computes accuracy and confusion matrix on the test set.
    """
    print(f"Evaluating QD model from log file: {log_file}")
    logreaders = init_logreaders(log_files=log_file)
    reader = logreaders[0]
    utils.set_seed(reader.params["seed"])
    params = reader.params
    qd_dataset = QDDataset(
        csv_file="ALL", transform=v2.Compose([LinesToImage()]))
    qd_dataset.class_labels = sorted(list(set(qd_dataset.labels)))
    train_frac = float(params.get("train_split", 0.8))
    val_frac = float(params.get("val_split", 0.1))
    if train_frac + val_frac > 1.0:
        raise ValueError("train_split + val_split must be <= 1.0")
    total = len(qd_dataset)
    train_size = int(total * train_frac)
    val_size = int(total * val_frac)
    test_size = total - train_size - val_size

    gen = torch.Generator().manual_seed(int(params.get("seed", 0)))
    train_dataset, val_dataset, test_dataset = random_split(
        qd_dataset, [train_size, val_size, test_size], generator=gen
    )

    batch_size = int(params.get("batch_size", 32))
    num_workers = int(params.get("num_workers", 4))
    pin_memory = bool(params.get("gpu", False))

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    model_name = reader.params["best_model_filename"]
    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Model file '{model_name}' does not exist.")
    device = torch.device(
        "cuda" if reader.params["gpu"] else "cpu")
    model = get_model(reader.params)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        acc, cm = check_accuracy(loader=test_dataloader,
                                 model=model,
                                 device=device,  # type: ignore
                                 class_labels=qd_dataset.class_labels,
                                 conf_matrix=True)

        cm_df = pd.DataFrame(
            cm, index=[f"True: {cls}" for cls in qd_dataset.class_labels],
            columns=[f"Pred: {cls}" for cls in qd_dataset.class_labels])
    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print("Class Labels:", qd_dataset.class_labels)


def create_example_imaes():
    data_dir = str(PathsConfig.data_dir)

    cls_name = ["sun", "wheel",  "pear", "onion", "mushroom", "hot air balloon",
                "cookie",  "apple",  "carrot", "boomerang"]
    # input_filename = data_dir + "/QuickDraw/"+cls_name+".ndjson"  # type: ignore
    canvas = np.empty((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), dtype=np.uint8)

    i_max = 10
    fig, axs = plt.subplots(len(cls_name), i_max,
                            figsize=(i_max*4, len(cls_name)*4))
    for j, cls in enumerate(cls_name):
        i = 0
        print(f"Processing class: {cls}, {j}")
        input_filename = data_dir + "/QuickDraw/"+cls+".ndjson"  # type: ignore
        with open(input_filename, 'r', encoding='utf-8') as infile:
            for line in infile:
                if i < i_max:
                    obj_line = line.strip()
                    ink, label = parse_line(obj_line)
                    canvas = render_ink_to_canvas(ink)

                    if isinstance(canvas, torch.Tensor):
                        img = canvas.detach().cpu().numpy()
                    else:
                        img = np.array(canvas)

                    if img.ndim == 3 and img.shape[0] in (1, 3):
                        if img.shape[0] == 1:
                            img = img[0]
                            cmap = 'gray'
                        else:
                            img = np.transpose(img, (1, 2, 0))
                            cmap = None
                    else:
                        cmap = 'gray' if img.ndim == 2 else None

                    axs[j, i].imshow(img, cmap=cmap, vmin=0, vmax=1)
                    axs[j, i].set_title(label if 'label' in locals() else '')
                    axs[j, i].axis('off')
                    i += 1

    fig.savefig(f"{_REPO_ROOT}/IB_DL/images/quickdraw.png")


def create_example_imgs_1_pc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7
    })
    data_dir = str(PathsConfig.data_dir)
    cls_name = ["apple", "boomerang", "carrot", "cookie",
                "hot air balloon", "mushroom", "onion", "pear", "sun", "wheel"]

    fig, axs = plt.subplots(2, 5, figsize=(3.5, 2))
    axs = axs.flatten()

    for j, cls in enumerate(cls_name):
        print(f"Processing class: {cls}")
        input_filename = data_dir + "/QuickDraw/" + cls + ".ndjson"
        with open(input_filename, 'r', encoding='utf-8') as infile:
            for idx, line in enumerate(infile):
                # Parameter for which image to take (0-indexed)
                image_index = 3
                if idx == image_index:  # Take the specified image from each class
                    obj_line = line.strip()
                    ink, label = parse_line(obj_line)
                    canvas = render_ink_to_canvas(ink)

                    if isinstance(canvas, torch.Tensor):
                        img = canvas.detach().cpu().numpy()
                    else:
                        img = np.array(canvas)

                    if img.ndim == 3 and img.shape[0] in (1, 3):
                        if img.shape[0] == 1:
                            img = img[0]
                            cmap = 'gray'
                        else:
                            img = np.transpose(img, (1, 2, 0))
                            cmap = None
                    else:
                        cmap = 'gray' if img.ndim == 2 else None

                    # axs[j].imshow(img, cmap=cmap, vmin=0, vmax=1)
                    img_inverted = 1 - img
                    axs[j].imshow(img_inverted, cmap=cmap, vmin=0, vmax=1)
                    axs[j].set_title(label)
                    # axs[j].axis('off')
                    axs[j].spines['top'].set_visible(True)
                    axs[j].spines['right'].set_visible(True)
                    axs[j].spines['bottom'].set_visible(True)
                    axs[j].spines['left'].set_visible(True)
                    axs[j].set_xticks([])
                    axs[j].set_yticks([])
                    break

    fig.savefig(
        f"{_REPO_ROOT}/IB_DL/images/quickdraw_1pc.pdf")
    plt.close()


if __name__ == "__main__":

    # 1. OPTIONAL: Generate visualization dataset samples
    # create_example_imgs_1_pc()

    # 2. PRETRAINING FOR TABLE 3
    # The two checkpoints behind the Quick, Draw! rows of Table 3 (random and
    # ImageNet initialisation), with the settings logged by the published runs.
    # models.py loads them from result/QD_resnet50.pt and
    # result/QD_resnet50_pretrained.pt (the 'QD_resnet50' and
    # 'QD_pretrained_resnet50' models of the pre_train experiment).

    download_quickdraw()
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    for model in ["resnet50", "resnet50_pretrained"]:
        prefix = f"QD_{model}"
        suffix = f"_{date.today().isoformat()}"

        train_params = {
            "model": model,
            "optimizer": "SGD",
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "output_dir": output_dir,
            "log_level": "INFO",
            "seed": 42,
            "gpu": torch.cuda.is_available(),
            "log_filename": os.path.join(output_dir, f"{prefix}_log_{suffix}.log"),
            "prefix": prefix,
            "suffix": suffix,
            "num_workers": 12,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "output_items": ['loss_plot', 'acc_plot', 'conf_matrix', 'cams'],
            "early_stop": True,
            "early_counter": 3,
            "eval_interval": 3,
            "class_labels": QD_CLASSES,
            "test_aggregation": 'all',
            "transformations": [{"name": "LinesToImage"}],
            "best_model_filename": os.path.join(output_dir, f"{prefix}.pt"),
            "image_size": DEFAULT_IMG_SIZE,
            "lw": LINE_WIDTH
        }

        print(f"\n--- Starting QuickDraw {model} pretraining ---")
        pretrain_qd(train_params)

    # 3. EXAMPLE EVALUATION BLOCK
    # eval_qd_model(log_file=f"{_REPO_ROOT}/IB_DL/SIN_QD/results/QD_resnet18_log__2025-11-25.log")
