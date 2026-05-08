import ast
from email.header import Header
from pathlib import Path
from typing import Sequence
import math
import random
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import pandas as pd

from IB_attention.logger import LogReader

### GLOBAL VARIABLES ###
LOG_LEVEL = "INFO"
raw_df = None
MEASURENT_FILES = ["all_measurements_1.csv",
                   "all_measurements_2.csv",
                   "all_measurements_3.csv"]

# Corner values for absolute scale loop representation
R_MAX = 7.0
R_MIN = 0
X_MAX = 0.5
X_MIN = -7.0
F_MAX = 1.5
F_MIN = -1.5
V_MIN = -0.2
V_MAX = 1.2

# ANSI escape codes for colors and styles
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# Foreground colors
RED_FG = "\033[31m"
GREEN_FG = "\033[32m"
YELLOW_FG = "\033[33m"
BLUE_FG = "\033[34m"
MAGENTA_FG = "\033[35m"
CYAN_FG = "\033[36m"
WHITE_FG = "\033[37m"
BLACK_FG = "\033[30m"
DEFAULT_FG = "\033[39m"  # Resets foreground to default

# Background colors (highlighted)
RED_BG = "\033[41m"
GREEN_BG = "\033[42m"
YELLOW_BG = "\033[43m"
BLUE_BG = "\033[44m"
MAGENTA_BG = "\033[45m"
CYAN_BG = "\033[46m"
WHITE_BG = "\033[47m"
DEFAULT_BG = "\033[49m"  # Resets background to default

# EPOCH PRINT COLORING HELPERS
header = True
epoch_counter = 0
tloss = 10000
vloss = 10000
tacc = 0.0
vacc = 0.0

# VISUALIZATION LABELS
ZRS_LABEL = r"$\mathrm{Zrs \ (hPa \cdot s \cdot L^{-1})}$"
VDOT_LABEL = r"$\mathrm{\dot{V}} (L/s)$"
V_LABEL = r"$\mathrm{V} \ (L)$"


RAINBOW_COLOR_MAP = [
    "\033[31m",  # Red
    "\033[91m",  # Bright Red (for gradation)
    "\033[93m",  # Bright Yellow
    "\033[33m",  # Yellow (used as Orange, closest ANSI)
    "\033[92m",  # Bright Green
    "\033[32m",  # Green
    "\033[96m",  # Bright Cyan (for gradation)
    "\033[36m",  # Cyan (used as Blue-Green)
    "\033[34m",  # Blue
    "\033[94m",  # Bright Blue
    "\033[95m",  # Bright Magenta (Violet)
    "\033[35m",  # Magenta (used as Indigo/Violet)
    "\033[90m",  # Bright Black (Gray, for extra gradation)
]

COLOR_MAP = {
    "red": (RED_FG, "", ""),
    "blue": (BLUE_FG, "", ""),
    "yellow": (YELLOW_FG, "", ""),
    "base": (DEFAULT_FG, "", ""),
    "default": (DEFAULT_FG, "", ""),
    "magenta": (MAGENTA_FG, "", ""),
    "cyan": (CYAN_FG, "", ""),
    "green": (GREEN_FG, "", ""),
    "black": (BLACK_FG, "", ""),
    "highlighted_yellow": (BOLD + BLACK_FG, YELLOW_BG, ""),
    "highlighted_red": (WHITE_FG, RED_BG, ""),
    "highlighted_green": (BOLD + BLACK_FG, GREEN_BG, ""),
    "highlighted_cyan": (BOLD + BLACK_FG, CYAN_BG, ""),
    "bold_red": (BOLD + RED_FG, "", ""),
    "underline_blue": (UNDERLINE + BLUE_FG, "", ""),
    "error": (RED_FG, "", ""),
    "warning": (YELLOW_FG, "", ""),
}

STYLE_DICT_DARK = {
    0:  {'color': "#010F1A", 'linestyle': '-', 'marker': None},
    1:  {'color': "#7e4009", 'linestyle': '-', 'marker': None},
    2:  {'color': "#0c3f0c", 'linestyle': '-', 'marker': None},
    3:  {'color': "#3F0B0B", 'linestyle': '-', 'marker': None},
}

STYLE_DICT = {
    0:  {'color': "#044c80", 'linestyle': '-', 'marker': None},  # Muted Blue
    1:  {'color': '#ff7f0e', 'linestyle': '-', 'marker': None},  # Safety Orange
    2:  {'color': '#2ca02c', 'linestyle': '-', 'marker': None},  # Cooked Asparagus Green
    3:  {'color': '#d62728', 'linestyle': '-', 'marker': None},  # Brick Red
    4:  {'color': "#bb80f1", 'linestyle': '-', 'marker': None},  # Muted Purple
    5:  {'color': "#4b2b25", 'linestyle': '-', 'marker': None},  # Chestnut Brown
    6:  {'color': "#d425a0", 'linestyle': '-', 'marker': None},  # Raspberry Yogurt Pink
    7:  {'color': '#7f7f7f', 'linestyle': '-', 'marker': None},  # Middle Gray
    8:  {'color': "#eeee0e", 'linestyle': '-', 'marker': None},  # Curry Yellow-Green
    9:  {'color': '#17becf', 'linestyle': '-', 'marker': None},  # Blue-Teal

    10:  {'color': "#044c80", 'linestyle': '--', 'marker': None},  # Muted Blue
    11:  {'color': '#ff7f0e', 'linestyle': '--', 'marker': None},  # Safety Orange
    12:  {'color': '#2ca02c', 'linestyle': '--', 'marker': None},  # Cooked Asparagus Green
    13:  {'color': '#d62728', 'linestyle': '--', 'marker': None},  # Brick Red
    14:  {'color': "#bb80f1", 'linestyle': '--', 'marker': None},  # Muted Purple
    15:  {'color': "#4b2b25", 'linestyle': '--', 'marker': None},  # Chestnut Brown
    16:  {'color': "#d425a0", 'linestyle': '--', 'marker': None},  # Raspberry Yogurt Pink
    17:  {'color': '#7f7f7f', 'linestyle': '--', 'marker': None},  # Middle Gray
    18:  {'color': "#eeee0e", 'linestyle': '--', 'marker': None},  # Curry Yellow-Green
    19:  {'color': '#17becf', 'linestyle': '--', 'marker': None},  # Blue-Teal

    20:  {'color': "#044c80", 'linestyle': ':', 'marker': None},  # Muted Blue
    21:  {'color': '#ff7f0e', 'linestyle': ':', 'marker': None},  # Safety Orange
    22:  {'color': '#2ca02c', 'linestyle': ':', 'marker': None},  # Cooked Asparagus Green
    23:  {'color': '#d62728', 'linestyle': ':', 'marker': None},  # Brick Red
    24:  {'color': "#bb80f1", 'linestyle': ':', 'marker': None},  # Muted Purple
    25:  {'color': "#4b2b25", 'linestyle': ':', 'marker': None},  # Chestnut Brown
    26:  {'color': "#d425a0", 'linestyle': ':', 'marker': None},  # Raspberry Yogurt Pink
    27:  {'color': '#7f7f7f', 'linestyle': ':', 'marker': None},  # Middle Gray
    28:  {'color': "#eeee0e", 'linestyle': ':', 'marker': None},  # Curry Yellow-Green
    29:  {'color': '#17becf', 'linestyle': ':', 'marker': None},  # Blue-Teal

    30:  {'color': "#044c80", 'linestyle': '-.', 'marker': None},  # Muted Blue
    31:  {'color': '#ff7f0e', 'linestyle': '-.', 'marker': None},  # Safety Orange
    32:  {'color': '#2ca02c', 'linestyle': '-.', 'marker': None},  # Cooked Asparagus Green
    33:  {'color': '#d62728', 'linestyle': '-.', 'marker': None},  # Brick Red
    34:  {'color': "#bb80f1", 'linestyle': '-.', 'marker': None},  # Muted Purple
    35:  {'color': "#4b2b25", 'linestyle': '-.', 'marker': None},  # Chestnut Brown
    36:  {'color': "#d425a0", 'linestyle': '-.', 'marker': None},  # Raspberry Yogurt Pink
    37:  {'color': '#7f7f7f', 'linestyle': '-.', 'marker': None},  # Middle Gray
    38:  {'color': "#eeee0e", 'linestyle': '-.', 'marker': None},  # Curry Yellow-Green
    39:  {'color': '#17becf', 'linestyle': '-.', 'marker': None},  # Blue-Teal

    40: {'color': "#0B0227", 'linestyle': '-', 'marker': None},  # Deep Indigo
    41: {'color': "#f884df", 'linestyle': '-', 'marker': None},  # Light Orchid
    42: {'color': "#193b03", 'linestyle': '-', 'marker': None},  # Leaf Green
    43: {'color': "#554003", 'linestyle': '-', 'marker': None},  # Golden Yellow
    44: {'color': "#0D0AA3", 'linestyle': '-', 'marker': None},  # Bright Royal Blue
    45: {'color': "#f3ad7e", 'linestyle': '-', 'marker': None},  # Sunset Orange
    46: {'color': "#a5a5a5", 'linestyle': '-', 'marker': None},  # Silver Gray
    47: {'color': "#603a7c", 'linestyle': '-', 'marker': None},  # Deep Violet
    48: {'color': "#85ff33", 'linestyle': '-', 'marker': None},  # Hot Pink
    49: {'color': "#0f8392", 'linestyle': '-', 'marker': None},  # Sky Cyan

    50: {'color': "#0B0227", 'linestyle': '--', 'marker': None},  # Deep Indigo
    51: {'color': "#f884df", 'linestyle': '--', 'marker': None},  # Light Orchid
    52: {'color': "#193b03", 'linestyle': '--', 'marker': None},  # Leaf Green
    53: {'color': "#554003", 'linestyle': '--', 'marker': None},  # Golden Yellow
    54: {'color': "#0D0AA3", 'linestyle': '--', 'marker': None},  # Royal Blue
    55: {'color': "#f3ad7e", 'linestyle': '--', 'marker': None},  # Sunset Orange
    56: {'color': "#a5a5a5", 'linestyle': '--', 'marker': None},  # Silver Gray
    57: {'color': "#603a7c", 'linestyle': '--', 'marker': None},  # Deep Violet
    58: {'color': "#85ff33", 'linestyle': '--', 'marker': None},  # Hot Pink
    59: {'color': "#0f8392", 'linestyle': '--', 'marker': None},  # Sky Cyan

    60: {'color': "#0B0227", 'linestyle': ':', 'marker': None},  # Deep Indigo
    61: {'color': "#f884df", 'linestyle': ':', 'marker': None},  # Light Orchid
    62: {'color': "#193b03", 'linestyle': ':', 'marker': None},  # Leaf Green
    63: {'color': "#554003", 'linestyle': ':', 'marker': None},  # Golden Yellow
    64: {'color': "#0D0AA3", 'linestyle': ':', 'marker': None},  # Bright Royal Blue
    65: {'color': "#f3ad7e", 'linestyle': ':', 'marker': None},  # Sunset Orange
    66: {'color': "#a5a5a5", 'linestyle': ':', 'marker': None},  # Silver Gray
    67: {'color': "#603a7c", 'linestyle': ':', 'marker': None},  # Deep Violet
    68: {'color': "#85ff33", 'linestyle': ':', 'marker': None},  # Hot Pink
    69: {'color': "#0f8392", 'linestyle': ':', 'marker': None},  # Sky Cyan

    70: {'color': "#0B0227", 'linestyle': '-.', 'marker': None},  # Deep Indigo
    71: {'color': "#f884df", 'linestyle': '-.', 'marker': None},  # Light Orchid
    72: {'color': "#193b03", 'linestyle': '-.', 'marker': None},  # Leaf Green
    73: {'color': "#554003", 'linestyle': '-.', 'marker': None},  # Golden Yellow
    74: {'color': "#0D0AA3", 'linestyle': '-.', 'marker': None},  # Bright Royal Blue
    75: {'color': "#f3ad7e", 'linestyle': '-.', 'marker': None},  # Sunset Orange
    76: {'color': "#a5a5a5", 'linestyle': '-.', 'marker': None},  # Silver Gray
    77: {'color': "#603a7c", 'linestyle': '-.', 'marker': None},  # Deep Violet
    78: {'color': "#85ff33", 'linestyle': '-.', 'marker': None},  # Hot Pink
    79: {'color': "#0f8392", 'linestyle': '-.', 'marker': None},  # Sky Cyan
}

### FUNCTIONS ###


def cprint(text: str, color_identifier: str = "default", end: str = "\n"):
    """
    Prints the given text to the terminal with specified color and/or highlighting.

    Parameters:
        color_identifier (str): A string identifying the desired color/highlight.
                                Valid options include: "red", "blue", "yellow", "base",
                                "default", "magenta", "cyan", "highlighted_yellow",
                                "highlighted_red", "highlighted_green", "highlighted_cyan".
                                Case-insensitive.
        text (str): The string content to be printed.
        end (str, optional): The string appended after the last value, default a newline.
    """
    color_identifier_lower = color_identifier.lower()

    fg_code, bg_code, style_code = COLOR_MAP.get(
        color_identifier_lower, (DEFAULT_FG, "", ""))

    if LOG_LEVEL != "NOTHING":
        print(f"{style_code}{fg_code}{bg_code}{text}{RESET}", end=end)


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def store_raw_data(params):
    global raw_df
    if raw_df is None:
        if params["all_measurements"]:
            file_paths = [os.path.join(
                params["data_dir"],
                f"all_measurements_{i}.csv") for i in [1, 2, 3]]
        else:
            file_paths = [os.path.join(params["data_dir"], "loops_filtered.csv")]

        df_list = []
        for path in file_paths:
            df_list.append(pd.read_csv(path, nrows=params.get("num_rows")))

        raw_df = pd.concat(df_list, ignore_index=True)
    return raw_df.copy()


def check_param_compatibility(params: dict):
    if params["model"] == "baseline" and {
            "name": "Grayscale"} != params["transformations"][
            -1]:
        params["transformations"].append({"name": "Grayscale"})

    if params["model"] == "cnn_scales" and {
            "name": "AddScale"} != params["transformations"][
            -1]:
        params["transformations"].append({"name": "AddScale"})

    for t in params["transformations"]:
        if t["name"] == "ToMeasurements":
            for t2 in params["transformations"]:
                if t2["name"] == "LoopToIBFeatures":
                    t2["measurement"] = True
    if params["model"] == "fc_cc":
        params["transformations"].append({"name": "Flatten"})


def init_logreaders(log_files: list[str | Path] | str | Path) -> list[LogReader]:
    """
    Initialize log readers from the provided log files.
    Labels are optional and can be used for differentiating experiments when plotting.
    If a single log file is provided, it is converted to a list.
    If a single label is provided, it is converted to a list.
    """
    if isinstance(log_files, str) or isinstance(log_files, Path):
        log_files = [log_files]

    logreaders: list[LogReader] = []
    for file in log_files:
        if not os.path.exists(file):
            print(f"Log file {file} does not exist.")
            continue
        logreader = LogReader(file)
        logreader.read()
        logreader.read_params()
        logreaders.append(logreader)
    return logreaders


def print_epoch(
    losses: dict[str, float],
    accuracies: dict[str, float],
    timing: float,
    epoch_id: int,
    lr: float,
    early_counter: int,
    saved: bool,
    print_level: str = "info"
):
    """
    Prints the epoch information including losses, accuracies, timing, learning rate, early stopping counter, and whether the model was saved.

    Args:
        losses (list[float]): List of loss values for the epoch.
        accuracies (list[float]): List of accuracy values for the epoch.
        timing (list[float]): List of timing values for the epoch.
        epoch_id (int): The current epoch number.
        lr (float): The learning rate used in this epoch.
        early_counter (int): The early stopping counter value.
        saved (bool): Whether the model was saved during this epoch.
        print_level (str): The level of detail to print ("info", "debug", etc.).
    """
    global header, epoch_counter, tloss, vloss, tacc, vacc
    if header:  # type: ignore
        cprint(f"{'Epoch':<12}", "default", end="")
        cprint(f"{'T Loss':<12}", "default", end="")
        cprint(f"{'V Loss':<12}", "default", end="")
        cprint(f"{'T Acc':<12}", "default", end="")
        cprint(f"{'V Acc':<12}", "default", end="")
        cprint(f"{'Time':<12}", "default", end="")
        cprint(f"{'Saved':<12}", "default", end="")
        cprint(f"{'LR':<12}", "default", end="")
        cprint(f"{'Early':<12}", "default", end="\n")
        header = False

    def color_by_improvement(curr, prev, acc=True):
        """ Determines the color based on improvement from previous value. 1% change threshold.
        Args:
            curr (float): Current value to compare.
            prev (float): Previous value to compare against.
            higher_is_better (bool): If True, higher values are better; otherwise, lower values are better.
        Returns:
            str: ANSI color code for the text."""
        if curr < 0:
            return "black"
        if prev == 10000 or prev == 0.0:
            return "default"
        percent_change = curr-prev

        if percent_change == 0:
            return "default"
        if acc:
            return "green" if curr > prev else "default"
        else:
            return "green" if curr < prev else "default"

    if print_level == "INFO":
        t_loss_color = color_by_improvement(
            losses['Train'], tloss, acc=False)
        v_loss_color = color_by_improvement(
            losses['Val'], vloss, acc=False)
        t_acc_color = color_by_improvement(
            accuracies['Train'], tacc, acc=True)
        v_acc_color = color_by_improvement(
            accuracies['Val'], vacc, acc=True)

        # Use rainbow color for epoch_id
        rainbow_color = RAINBOW_COLOR_MAP[epoch_id % len(RAINBOW_COLOR_MAP)]
        print(f"{rainbow_color}{epoch_id:<12}{RESET}", end="")
        cprint(f"{losses['Train']:<12.3f}", t_loss_color, end="")
        cprint(f"{losses['Val']:<12.3f}", v_loss_color, end="")
        if accuracies['Train'] == -1.0:
            cprint(f"{'-':<12}", "black", end="")
            cprint(f"{'-':<12}", "black", end="")
        else:
            cprint(f"{accuracies['Train']:<12.3f}", t_acc_color, end="")
            cprint(f"{accuracies['Val']:<12.3f}", v_acc_color, end="")
        cprint(f"{timing:<12.2f}", "default", end="")
        if saved:
            cprint(f"{'✓':<12}", "cyan", end="")
        else:
            cprint(f"{'-':<12}", "default", end="")
        cprint(f"{lr:<12.5f}", "default", end="")
        cprint(f"{early_counter:<12}", "default", end="\n")

        # Update global trackers
        if tloss > losses['Train']:
            tloss = losses['Train']
        if vloss > losses['Val']:
            vloss = losses['Val']
        if accuracies['Train'] != -1.0:
            if tacc < accuracies['Train']:
                tacc = accuracies['Train']
            if vacc < accuracies['Val']:
                vacc = accuracies['Val']


def visualize_tensor_data(
        train_features: torch.FloatTensor, train_labels: tuple[str],
        flattened: bool = True,
        save_path: str = "IB_attention/images/batch_tensor_visualization.png"):
    """Visualizes a batch of data by their labels.

    Args:
    train_features (torch.FloatTensor): Loop data stored in Tensors.
    train_labels (tuple[str]): Labels
    flattened (bool, optional): _description_. Defaults to True.
    """
    labels = set(train_labels)
    num_classes = len(labels)
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {num_classes}")

    label_codes = {key: i for i, key in enumerate(sorted(labels))}
    fig, ax = plt.subplots(ncols=num_classes, figsize=(num_classes*4, 4))

    for i in range(len(train_features)):
        if flattened:
            flow_data = train_features[i, 0:50]
            resistance_data = train_features[i, 100:150]
            reactance_data = train_features[i, 150:200]
        else:
            flow_data = train_features[i, 0, :]
            resistance_data = train_features[i, 2, :]
            reactance_data = train_features[i, 3, :]
        ax[label_codes[train_labels[i]]].plot(flow_data, resistance_data, c='red',
                                              label="Rrs", linewidth=1, alpha=0.8)
        ax[label_codes[train_labels[i]]].plot(flow_data, reactance_data, c='blue',
                                              label="Xrs", linewidth=1, alpha=0.8)
        for i, key in enumerate(sorted(labels)):
            ax[label_codes[key]].axvline(
                0, color='gray', linestyle='--', alpha=0.5)
            ax[label_codes[key]].set_xlabel("Flow (L·s⁻¹)")
            ax[label_codes[key]].set_ylabel(
                "Zrs (hPa·s·L⁻¹)")
            ax[label_codes[key]].set_title(key)
    plt.tight_layout()
    plt.savefig(save_path)
    print("The result is saved in IB_attention/images/batch_tensor_visualization.png")


def visualize_image_data(
        image_batch: torch.Tensor,
        labels: Sequence[str],
        save_path: str = "IB_attention/images/batch_visualization_image.png"):
    """Visualizes each image in a batch individually in subplots.

    Args:
        image_batch (torch.Tensor): Batch of images, shape (B, C, H, W).
        labels (Sequence[str]): List of labels (str) for each image.
        save_path (str, optional): Path to save the figure.
    """
    batch_size = image_batch.size(0)
    cols = 4
    rows = math.ceil(batch_size / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = axs.flatten()

    for i in range(batch_size):
        ax = axs[i]
        img = image_batch[i].permute(1, 2, 0).numpy()
        img = img.clip(0, 1)

        ax.imshow(img)
        ax.set_title(labels[i])
        ax.axis('off')

    for j in range(batch_size, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved individual image batch visualization to {save_path}")


def visualize_image_data_per_channel(
    image_batch: torch.Tensor,
    labels: Sequence[str],
    save_path: str = "IB_attention/images/batch_visualization_per_layer.png"
):
    """
    Visualizes each layer/channel of every image in the batch.
    Each row = one image in the batch.
    Each column = one channel of the image.

    Args:
        image_batch (torch.Tensor): Tensor of shape (B, C, H, W)
        labels (Sequence[str]): Labels for each image
        save_path (str, optional): Path to save the figure
    """
    batch_size, num_channels, H, W = image_batch.shape
    fig, axs = plt.subplots(batch_size, num_channels,
                            figsize=(num_channels * 3, batch_size * 3))

    if batch_size == 1:
        axs = axs[None, :]  # Add batch axis
    if num_channels == 1:
        axs = axs[:, None]  # Add channel axis

    for i in range(batch_size):
        for j in range(num_channels):
            ax = axs[i][j]
            img = image_batch[i, j].numpy()
            ax.imshow(img, cmap='gray', origin='upper')
            if i == 0:
                ax.set_title(f"Layer {j}")
            if j == 0:
                ax.set_ylabel(labels[i])
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved layer-wise batch visualization to {save_path}")


def get_optimizer(opt_name: str, nn_model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    """
    Factory function to get the optimizer based on the optimizer name.

    Args:
        opt_name (str): Name of the optimizer to retrieve.
        model (nn.Module): The model for which the optimizer is created.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: The optimizer instance.
    """
    if opt_name == "SGD":
        optimizer_ = torch.optim.SGD(
            nn_model.parameters(), lr=learning_rate)
    elif opt_name == "Adam":
        optimizer_ = torch.optim.Adam(
            nn_model.parameters(), lr=learning_rate)
    elif opt_name == "AdamW":
        optimizer_ = torch.optim.AdamW(
            nn_model.parameters(), lr=learning_rate)
    elif opt_name == "RMSprop":
        optimizer_ = torch.optim.RMSprop(
            nn_model.parameters(), lr=learning_rate)
    elif opt_name == "Adagrad":
        optimizer_ = torch.optim.Adagrad(
            nn_model.parameters(), lr=learning_rate)
    elif opt_name == "NAG":
        optimizer_ = torch.optim.SGD(
            nn_model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    return optimizer_


def calculate_f1(cm_str):
    """Calculates the F1 score from a confusion matrix string, used in the test_result.csv file. The confusion matrix is expected to be in the format of a string representation of a list of lists, e.g. "[[50, 10], [5, 35]]". The function parses this string, computes the true positives, false positives, and false negatives for each class, and then calculates the precision, recall, and F1 score for each class before returning the average F1 score across all classes."""
    try:
        # Converts "[1, 2], [3, 4]" string to [[1, 2], [3, 4]]
        matrix = ast.literal_eval("[" + cm_str + "]")
        cm = np.array(matrix)

        f1_scores = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            f1_scores.append(f1)
        return np.mean(f1_scores)
    except:
        return 0.0
