from scipy.interpolate import RegularGridInterpolator
import matplotlib.ticker as ticker
import random

import matplotlib
from matplotlib import ticker
from matplotlib.lines import Line2D
import os
import re
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import pandas as pd
from IB_DL import datasets
from IB_DL.config import PathsConfig
from IB_DL.utils import STYLE_DICT, STYLE_DICT_DARK, VDOT_LABEL, ZRS_LABEL, calculate_f1
import matplotlib.colors as colors
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator

PAGE_WIDTH_INCHES = 7.16  # IEEE 2-column width in inches
COLUMN_WIDTH_INCHES = 3.5  # IEEE single column width in inches
AGGREGATION_LABELS = {
    "loop_wise": "Loop-wise",
    "median_loop": "Median loop",
    "majority_vote": "Soft vote"
}


def init_plt():
    plt.rcParams.update({'axes.labelsize': 9,  # for axis labels
                         'xtick.labelsize': 8,  # for x-tick labels
                         'ytick.labelsize': 8,  # for y-tick labels
                         'legend.fontsize': 9,
                         'figure.titlesize': 10,
                         'axes.linewidth': 0.3,
                         # 'ytick.labelweight': 'bold',  # Make y-tick labels bold
                         # 'xtick.labelweight': 'bold',  # Make x-tick labels bold
                         'figure.titleweight': 'bold',
                         'figure.titlesize': 10,
                         'axes.titlesize': 10,
                        #  'axes.labelweight': 'bold',
                         'font.size': 10,
                         'font.family': 'serif',
                         'font.serif': ['DejaVu Serif', 'Times New Roman'],
                         'text.latex.preamble': r"\usepackage{newtxtext} \usepackage{newtxmath} \boldmath",
                         'font.serif': 'Times New Roman'})
    plt.rcParams['mathtext.fontset'] = 'stix'

    plt.rcParams['font.family'] = 'serif'


def compare_train_val(
    train_epochs: list[list[int]],
    train_val: list[list[float]],
    val_epochs: list[list[int]],
    val_val: list[list[float]],
    labels: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Epochs",
    ylabel: str = "Value",
):
    num_labels = len(labels) if labels else len(train_val)
    num_cols = 1
    plt.figure()
    if title:
        plt.suptitle(title)
    plt.subplot(121)
    plt.title("Train")
    for i in range(len(train_val)):
        plt.plot(
            train_epochs[i],
            train_val[i],
            label=labels[i] if labels else f"{i+1}", color=STYLE_DICT[i % 80]
            ['color'],
            linestyle=STYLE_DICT[i % 80]['linestyle'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=num_cols, fancybox=True)

    plt.subplot(122)
    plt.title("Val")
    for i in range(len(val_val)):
        plt.plot(
            val_epochs[i],
            val_val[i],
            label=labels[i] if labels else f"{i+1}", color=STYLE_DICT[i % 80]
            ['color'],
            linestyle=STYLE_DICT[i % 80]['linestyle'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.legend()
    # plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_conf_matrix(conf_matrix, class_labels, save_path: Optional[str] = None,):
    df_confusion_matrix = pd.DataFrame(conf_matrix, index=range(len(class_labels)),
                                       columns=range(len(class_labels)))

    idxs, cols = zip(*[(class_labels[i], class_labels[i])
                     for i in range(len(class_labels))])
    df_confusion_matrix.index = idxs  # type: ignore
    df_confusion_matrix.columns = cols

    df_percentages = df_confusion_matrix.div(
        df_confusion_matrix.sum(axis=1), axis=0) * 100
    df_percentages = df_percentages.fillna(0)

    annot = df_confusion_matrix.astype(
        str) + "\n" + df_percentages.round(1).astype(str).map(lambda x: f"${x}\\%$")
    fig, ax = plt.subplots(figsize=(3, 3))
    annot_array = annot.to_numpy()
    colors = [(1, 0.9, 0.9), (0.8, 0.2, 0.2)]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        class_labels, colors)
    sns.heatmap(df_percentages, annot=annot_array, fmt="",  cbar=False, cmap=custom_cmap,
                # cmap="coolwarm",, vmin=-600, vmax=100, ,  annot_kws={"color": "black"}
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    sns.diverging_palette(170, 180, s=150, center='dark', as_cmap=True)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_test_accuracies(
        all_acc, acc_labels, eval_labels, save_path: Optional[str] = None):
    all_acc = np.array(all_acc)
    num_models = len(set(eval_labels))
    num_acc_types = len(set(acc_labels))
    unique_eval_labels = np.unique(eval_labels, return_index=True)
    ordered_unique_eval_labels = [eval_labels[i]
                                  for i in sorted(unique_eval_labels[1])]
    if all_acc.ndim == 1:
        # Reshape if it's a 1D array
        all_acc = all_acc.reshape(num_models, num_acc_types)
    df = pd.DataFrame(
        all_acc, columns=acc_labels[:num_acc_types], index=ordered_unique_eval_labels)
    # plt.figure(figsize=(15, 35))
    num_rows = (len(df) + 2) // 3  # Calculate number of rows needed
    plt.figure(figsize=(5, num_rows * 3))  # Set figure size
    ax = sns.heatmap(df, annot=True, fmt=".2f", cbar=False,
                     cmap="RdBu_r", ax=plt.gca(), square=True)

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_matrix(
    data,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm_type: str = 'linear',  # Options: 'linear', 'log', 'power'
    gamma: float = 2.0         # Used if norm_type is 'power'
):
    """
    Helper function to plot and save a heatmap matrix with optional non-linear scaling.
    """
    plt.figure(figsize=(8, 8))

    # Define the normalization based on user preference
    if norm_type == 'log':
        # Note: LogNorm requires values > 0.
        # It expands the LOW end (e.g., 0.1 to 1.0)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm_type == 'power':
        # PowerNorm with gamma > 1 expands the HIGH end
        # This is likely what you want for 'high accuracy focus'
        norm = colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    else:
        norm = None  # Defaults to Linear

    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Accuracy (%)'},
        vmin=vmin,
        vmax=vmax,
        norm=norm
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # Save before showing, as show() can clear the figure in some backends
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_augmentation_matrix(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['p'] = df['model_label'].str.split('_').str[1]
    df['strength'] = df['model_label'].str.split('_').str[3]
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['p', 'strength'])[
            'accuracy'].mean().reset_index()
        avg_result = avg_df.pivot(
            index='strength', columns='p', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    'Transformation probability',
                    'Transformation strength',
                    f"{experiment_path}/augmentation_matrix_{agg_type}.png",
                    norm_type='power',
                    gamma=2.0
                    )


def plot_augmentation_matrix_3x3(
        path_a: str, path_b: str, path_c: str, output_name: str = "IEEE_3x3_Matrix.pdf"):
    # IEEE 2-column width
    fig_width = 5
    fig_height = 5
    init_plt()

    paths = [path_a, path_b, path_c]
    models = ["ResNet-50", "Model A", "Model B"]
    aggregation_types = ["loop_wise",  "majority_vote", "median_loop",]
    row_labels = ["Loop-wise", "Soft vote", "Median loop"]

    # Load and process data for all paths and aggregation types
    pivoted_data = []
    for path in paths:
        piv = []
        for agg in aggregation_types:
            df = pd.read_csv(f"{path}/test_results.csv")
            df['p'] = df['model_label'].str.split('_').str[1]
            df['strength'] = df['model_label'].str.split('_').str[3]
            df_subset = df[df['aggregation_method'] == agg]
            piv.append(df_subset.groupby(['strength', 'p'])[
                'accuracy'].mean().unstack())
        pivoted_data.append(piv)

    # Calculate v_limits by aggregation method
    v_limits = []
    for row in range(3):
        row_min = min(pivoted_data[0][row].min().min(
        ), pivoted_data[1][row].min().min(), pivoted_data[2][row].min().min())
        row_max = max(pivoted_data[0][row].max().max(
        ), pivoted_data[1][row].max().max(), pivoted_data[2][row].max().max())
        v_limits.append((row_min, row_max))
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(3, 4, width_ratios=[
                          1, 1, 1, 0.08], hspace=0.2, wspace=0.2)

    for row in range(3):
        for col in range(3):
            df = pd.read_csv(f"{paths[col]}/test_results.csv")
            df['p'] = df['model_label'].str.split('_').str[1]
            df['strength'] = df['model_label'].str.split('_').str[3]
            df_subset = df[df['aggregation_method'] == aggregation_types[row]]
            pivoted_data = df_subset.groupby(['strength', 'p'])[
                'accuracy'].mean().unstack()
            ax = fig.add_subplot(gs[row, col])
            norm = colors.PowerNorm(
                gamma=2.0, vmin=v_limits[row][0], vmax=v_limits[row][1])
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
            sm.set_array([])
            sns.heatmap(pivoted_data,
                        annot=False,
                        cmap="coolwarm",
                        square=True,
                        vmin=v_limits[row][0], vmax=v_limits[row][1], norm=norm,
                        cbar=False,
                        ax=ax)

            ax.tick_params(axis='both', which='both', length=0)

            if row == 0:
                ax.set_title(models[col],
                             fontweight='bold', family='serif')

            if col == 0:
                ax.set_ylabel(f"Transformation strength", family='serif')
                ax.annotate(
                    AGGREGATION_LABELS[aggregation_types[row]],
                    xy=(-0.5, 0.5),
                    xycoords='axes fraction', rotation=90, ha='center', va='center',
                    fontweight='bold')
            else:
                ax.set_ylabel("")

            if row == 2:
                ax.set_xlabel("Probability", family='serif')
            else:
                ax.set_xlabel("")

        # Add colorbar for this row
        cbar_ax = fig.add_subplot(gs[row, 3])
        norm = colors.PowerNorm(
            gamma=2.0, vmin=v_limits[row][0], vmax=v_limits[row][1])
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=100, decimals=0))
        cbar.ax.tick_params(labelsize=9, length=0)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)
    # Add row labels
    # for row in range(3):
    #     fig.text(0.02, 0.83 - row * 0.32, row_labels[row],
    #              ha='right', va='center',  fontweight='bold', rotation=90)
    plt.savefig(output_name, format='pdf', bbox_inches='tight', dpi=300)
    print(f"3x3 plot saved as {output_name}")
    plt.close()


def upsample_grid(df_pivoted, p: int = 3):
    """
    Interpolates an NxN DataFrame into an (N*p)x(N*p) numpy array.
    Original values become the center of pxp blocks.

    Returns:
        interpolated_data: The upsampled 2D numpy array.
        locs: The 1D array of the original center coordinates in the new grid.
    """
    data = df_pivoted.values
    N = data.shape[0]  # Original grid dimension (e.g., 5)

    # Calculate the exact physical centers of the pxp blocks in the new grid
    # For p=3, locs = [1., 4., 7., 10., 13.]
    # For p=4, locs = [1.5, 5.5, 9.5, 13.5, 17.5]
    locs = np.arange(N) * p + (p - 1) / 2.0

    # Set up 2D linear interpolation (fill_value=None enables extrapolation at edges)
    interp = RegularGridInterpolator(
        (locs, locs), data, method='linear', bounds_error=False, fill_value=None)

    # Create the new high-resolution coordinate grid
    new_size = N * p
    y_new = np.arange(new_size)
    x_new = np.arange(new_size)
    Y, X = np.meshgrid(y_new, x_new, indexing='ij')

    # Evaluate the interpolated points
    interpolated_data = interp((Y, X))

    # Clip to 0-100% just in case extrapolation at the very edges overshoots
    return np.clip(interpolated_data, 0, 100), locs


def plot_augmentation_matrix_3x3_v2(
        path_a: str, path_b: str, path_c: str, output_name: str = "IEEE_3x3_Matrix.pdf",
        p: int = 3):
    # IEEE 2-column width
    fig_width = 4
    fig_height = 4
    init_plt()

    paths = [path_a, path_b, path_c]
    models = ["ResNet-50", "Model A", "Model B"]
    aggregation_types = ["loop_wise",  "majority_vote", "median_loop"]
    row_labels = ["Loop-wise", "Soft vote", "Median loop"]

    # 1. Load, process, and upsample data for global v_limits
    upsampled_data_grid = []
    for path in paths:
        path_data = []
        for agg in aggregation_types:
            df = pd.read_csv(f"{path}/test_results.csv")
            df['p'] = df['model_label'].str.split('_').str[1]
            df['strength'] = df['model_label'].str.split('_').str[3]
            df_subset = df[df['aggregation_method'] == agg]

            pivoted = df_subset.groupby(['strength', 'p'])[
                'accuracy'].mean().unstack()

            # Upsample dynamically using the parameter p
            upsampled, center_locs = upsample_grid(pivoted, p=p)

            path_data.append({
                'upsampled': upsampled,
                'center_locs': center_locs,
                'x_labels': pivoted.columns.tolist(),
                'y_labels': pivoted.index.tolist()
            })
        upsampled_data_grid.append(path_data)

    # Calculate v_limits by aggregation method (row) based on upsampled data
    v_limits = []
    for row in range(3):
        row_min = min(upsampled_data_grid[0][row]['upsampled'].min(),
                      upsampled_data_grid[1][row]['upsampled'].min(),
                      upsampled_data_grid[2][row]['upsampled'].min())
        row_max = max(upsampled_data_grid[0][row]['upsampled'].max(),
                      upsampled_data_grid[1][row]['upsampled'].max(),
                      upsampled_data_grid[2][row]['upsampled'].max())
        v_limits.append((row_min, row_max))

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(3, 4, width_ratios=[
                          1, 1, 1, 0.08], hspace=0.2, wspace=0.2)

    for row in range(3):
        for col in range(3):
            # Retrieve the pre-calculated array, centers, and original labels
            data_dict = upsampled_data_grid[col][row]
            plot_data = data_dict['upsampled']
            locs = data_dict['center_locs']
            x_labels = data_dict['x_labels']
            y_labels = data_dict['y_labels']

            ax = fig.add_subplot(gs[row, col])
            norm = colors.PowerNorm(
                gamma=2.0, vmin=v_limits[row][0], vmax=v_limits[row][1])

            sns.heatmap(plot_data,
                        annot=False,
                        cmap="coolwarm",
                        square=True,
                        vmin=v_limits[row][0], vmax=v_limits[row][1], norm=norm,
                        cbar=False,
                        ax=ax)

            ax.tick_params(axis='both', which='both', length=0)

            # Map the ticks to the center of the 1st, 3rd, and 5th original blocks.
            # Seaborn places ticks at the center of the cells, so we add 0.5 to our locs.
            tick_positions = [locs[0] + 0.5, locs[2] + 0.5, locs[4] + 0.5]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels([x_labels[0], x_labels[2], x_labels[4]])

            ax.set_yticks(tick_positions)
            ax.set_yticklabels([y_labels[0], y_labels[2], y_labels[4]])

            # Set Titles on Top Row
            if row == 0:
                ax.set_title(models[col], fontweight='bold', family='serif')

            # Handle Y-axis labels and Aggregation Labels (Col 0 only)
            if col == 0:
                if row == 1:
                    ax.set_ylabel("Transformation strength", family='serif')
                else:
                    ax.set_ylabel("")

                # Assuming AGGREGATION_LABELS is defined in your global scope
                ax.annotate(
                    AGGREGATION_LABELS[aggregation_types[row]],
                    xy=(-0.5, 0.5),
                    xycoords='axes fraction', rotation=90, ha='center', va='center',
                    fontweight='bold')
            else:
                ax.set_ylabel("")

            # Handle X-axis labels (Row 2 only)
            if row == 2:
                if col == 1:
                    ax.set_xlabel("Probability", family='serif')
                else:
                    ax.set_xlabel("")
            else:
                ax.set_xlabel("")

        # Add colorbar for this row
        cbar_ax = fig.add_subplot(gs[row, 3])
        norm = colors.PowerNorm(
            gamma=3.0, vmin=v_limits[row][0], vmax=v_limits[row][1])
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)

        # Set ticks to Min, Mid, and Max
        vmin, vmax = v_limits[row][0], v_limits[row][1]
        mid_val = vmax-4
        mid_val2 = vmax - 12
        cbar.set_ticks([vmin, mid_val, mid_val2, vmax])

        cbar.ax.yaxis.set_major_formatter(
            ticker.PercentFormatter(xmax=100, decimals=0))
        cbar.ax.tick_params(labelsize=9, length=0)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1)
    plt.savefig(output_name, format='pdf', bbox_inches='tight', dpi=300)
    print(f"{p}x{p} smoothed plot saved as {output_name}")
    plt.close()


def plot_augmentation_matrix_1x4(path_a: str, path_b: str,
                                 output_name: str = "IEEE_Paired_Matrix.pdf"):

    # IEEE 2-column width. Height slightly increased to accommodate labels.
    fig_width = 7.16
    fig_height = 2.2

    init_plt()

    # Order: F1 T1, F2 T1, F1 T2, F2 T2
    configs = [
        (path_a, "loop_wise", "a)"),
        (path_b, "loop_wise", "b)"),
        (path_a, "majority_vote", "c)"),
        (path_b, "majority_vote", "d)")
    ]

    # Load and process data
    pivoted_data = []
    for path, agg, label in configs:
        df = pd.read_csv(f"{path}/test_results.csv")
        df['p'] = df['model_label'].str.split('_').str[1]
        df['strength'] = df['model_label'].str.split('_').str[3]
        df_subset = df[df['aggregation_method'] == agg]
        pivoted_data.append(df_subset.groupby(['strength', 'p'])[
                            'accuracy'].mean().unstack())

    # Type-wise Scaling
    v_limits = [
        (min(pivoted_data[0].min().min(), pivoted_data[1].min().min()),
         max(pivoted_data[0].max().max(), pivoted_data[1].max().max())),
        (min(pivoted_data[2].min().min(), pivoted_data[3].min().min()),
         max(pivoted_data[2].max().max(), pivoted_data[3].max().max()))
    ]

    # Create a 1x6 grid: Heatmap, Heatmap, Colorbar, Heatmap, Heatmap, Colorbar
    # Width ratios ensure the 4 heatmaps (ratio 1) are exactly the same width
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, 6, width_ratios=[
                          1, 1, 0.08, 1, 1, 0.08], wspace=0.5)

    # Mapping which data goes into which heatmap column
    # Heatmap indices in gs: 0, 1, 3, 4 | Colorbar indices: 2, 5
    gs_map = [0, 1, 3, 4]

    for i, gs_idx in enumerate(gs_map):
        ax = fig.add_subplot(gs[gs_idx])
        type_idx = 0 if i < 2 else 1
        v_min, v_max = v_limits[type_idx]

        # Define colorbar axis only on the 2nd and 4th heatmaps
        cbar_ax = None
        if i == 1:
            cbar_ax = fig.add_subplot(gs[2])
        elif i == 3:
            cbar_ax = fig.add_subplot(gs[5])
        norm = colors.LogNorm(vmin=v_min, vmax=v_max)

        sns.heatmap(pivoted_data[i],
                    annot=False,
                    cmap="coolwarm",
                    square=True,
                    vmin=v_min, vmax=v_max, norm=norm,
                    cbar=(cbar_ax is not None),
                    cbar_ax=cbar_ax,
                    ax=ax)

        # Style cleanup
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_title(configs[i][2], fontweight='bold', family='serif')
        # ax.set_xlabel("", family='serif')

        # Label logic: Only on the left-most plot of each pair
        if i == 0:
            ax.set_ylabel("Transformation strength", family='serif')
        else:
            ax.set_ylabel("")
        if i == 3:
            ax.set_xlabel("Probability", family='serif')
        else:
            ax.set_xlabel("")

        if cbar_ax:
            cbar_ax.tick_params(labelsize=6, length=0)

    # plt.tight_layout doesn't play well with fixed GridSpec ratios,
    # so we use subplots_adjust for fine-tuning margins
    fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.2)

    plt.savefig(output_name, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Uniform plot saved as {output_name}")
    plt.close()


def plot_architecture_matrix(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['fe_layers'] = df['model_label'].str.split('_').str[2]
    df['fc_layers'] = df['model_label'].str.split('_').str[5]
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['fe_layers', 'fc_layers'])[
            'accuracy'].mean().reset_index()
        avg_result = avg_df.pivot(
            index='fc_layers', columns='fe_layers', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    'Feature extractor layers',
                    'Fully connected layers',
                    f"{experiment_path}/architecture_matrix_{agg_type}.png")


def plot_architecture_matrix_max(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['fe_layers'] = df['model_label'].str.split('_').str[2]
    df['fc_layers'] = df['model_label'].str.split('_').str[5]
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['fe_layers', 'fc_layers'])[
            'accuracy'].max().reset_index()
        avg_result = avg_df.pivot(
            index='fc_layers', columns='fe_layers', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    'Feature extractor layers',
                    'Fully connected layers',
                    f"{experiment_path}/max_architecture_matrix_{agg_type}.png")


def plot_architecture_matrix_1x4(path_a: str, path_b: str,
                                 output_name: str = "IEEE_Paired_Matrix.pdf"):
    # IEEE 2-column width. Height slightly increased to accommodate labels.
    fig_width = 7.16
    fig_height = 2.2

    init_plt()

    # Order: F1 T1, F2 T1, F1 T2, F2 T2
    configs = [
        (path_a, "loop_wise", "a)"),
        (path_b, "loop_wise", "b)"),
        (path_a, "majority_vote", "c)"),
        (path_b, "majority_vote", "d)")
    ]

    # Load and process data
    pivoted_data = []
    for path, agg, label in configs:
        df = pd.read_csv(f"{path}/test_results.csv")
        df['fe_layers'] = df['model_label'].str.split('_').str[2]
        df['fc_layers'] = df['model_label'].str.split('_').str[5]
        df_subset = df[df['aggregation_method'] == agg]
        pivoted_data.append(df_subset.groupby(['fe_layers', 'fc_layers'])[
                            'accuracy'].mean().unstack())

    # Type-wise Scaling
    v_limits = [
        (min(pivoted_data[0].min().min(), pivoted_data[1].min().min()),
         max(pivoted_data[0].max().max(), pivoted_data[1].max().max())),
        (min(pivoted_data[2].min().min(), pivoted_data[3].min().min()),
         max(pivoted_data[2].max().max(), pivoted_data[3].max().max()))
    ]

    # Create a 1x6 grid: Heatmap, Heatmap, Colorbar, Heatmap, Heatmap, Colorbar
    # Width ratios ensure the 4 heatmaps (ratio 1) are exactly the same width
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, 6, width_ratios=[
                          1, 1, 0.08, 1, 1, 0.08], wspace=0.5)

    # Mapping which data goes into which heatmap column
    # Heatmap indices in gs: 0, 1, 3, 4 | Colorbar indices: 2, 5
    gs_map = [0, 1, 3, 4]

    for i, gs_idx in enumerate(gs_map):
        ax = fig.add_subplot(gs[gs_idx])
        type_idx = 0 if i < 2 else 1
        v_min, v_max = v_limits[type_idx]

        # Define colorbar axis only on the 2nd and 4th heatmaps
        cbar_ax = None
        if i == 1:
            cbar_ax = fig.add_subplot(gs[2])
        elif i == 3:
            cbar_ax = fig.add_subplot(gs[5])
        norm = colors.LogNorm(vmin=v_min, vmax=v_max)

        sns.heatmap(pivoted_data[i],
                    annot=False,
                    cmap="coolwarm",
                    square=True,
                    vmin=v_min, vmax=v_max, norm=norm,
                    cbar=(cbar_ax is not None),
                    cbar_ax=cbar_ax,
                    ax=ax)

        # Style cleanup
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_title(configs[i][2], fontweight='bold', family='serif')

        # Label logic: Only on the left-most plot of each pair
        if i == 0:
            ax.set_ylabel("FC Layers", family='serif')
        else:
            ax.set_ylabel("")

        if i >= 2:
            ax.set_xlabel("FE Layers", family='serif')
        else:
            ax.set_xlabel("")

        if cbar_ax:
            cbar_ax.tick_params(labelsize=6, length=0)

    # plt.tight_layout doesn't play well with fixed GridSpec ratios,
    # so we use subplots_adjust for fine-tuning margins
    fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.2)

    plt.savefig(output_name, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Uniform plot saved as {output_name}")
    plt.close()


def plot_architecture_matrix_1x6(path_a: str, path_b: str,
                                 output_name: str = "IEEE_Paired_Matrix.pdf"):
    # IEEE 2-column width. Height slightly increased to accommodate labels.
    fig_width = 7.16
    fig_height = 1.8

    init_plt()

    # Order: F1 T1, F2 T1, F1 T2, F2 T2
    configs = [
        (path_a, "loop_wise", "a)"),
        (path_b, "loop_wise", "b)"),
        (path_a, "majority_vote", "c)"),
        (path_b, "majority_vote", "d)"),
        (path_a, "median_loop", "e)"),
        (path_b, "median_loop", "f)")
    ]

    # Load and process data
    pivoted_data = []
    for path, agg, label in configs:
        df = pd.read_csv(f"{path}/test_results.csv")
        df['fe_layers'] = df['model_label'].str.split('_').str[2]
        df['fc_layers'] = df['model_label'].str.split('_').str[5]
        df_subset = df[df['aggregation_method'] == agg]
        pivoted_data.append(df_subset.groupby(['fe_layers', 'fc_layers'])[
                            'accuracy'].mean().unstack())

    # Type-wise Scaling
    v_limits = [
        (min(pivoted_data[0].min().min(), pivoted_data[1].min().min()),
         max(pivoted_data[0].max().max(), pivoted_data[1].max().max())),
        (min(pivoted_data[2].min().min(), pivoted_data[3].min().min()),
         max(pivoted_data[2].max().max(), pivoted_data[3].max().max())),
        (min(pivoted_data[4].min().min(), pivoted_data[5].min().min()),
            max(pivoted_data[4].max().max(), pivoted_data[5].max().max()))
    ]

    # Create a 1x6 grid: Heatmap, Heatmap, Colorbar, Heatmap, Heatmap, Colorbar
    # Width ratios ensure the 4 heatmaps (ratio 1) are exactly the same width
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(1, 11, width_ratios=[
                          1, 1, 0.08, 0.01, 1, 1, 0.08, 0.01, 1, 1, 0.08], wspace=0.5)

    # Mapping which data goes into which heatmap column
    # Heatmap indices in gs: 0, 1, 3, 4, 6, 7 | Colorbar indices: 2, 5, 8
    gs_map = [0, 1, 4, 5, 8, 9]

    for i, gs_idx in enumerate(gs_map):
        ax = fig.add_subplot(gs[gs_idx])
        type_idx = 0 if i < 2 else 1 if i < 4 else 2
        v_min, v_max = v_limits[type_idx]

        # Define colorbar axis only on the 2nd and 4th heatmaps
        cbar_ax = None
        if i == 1:
            cbar_ax = fig.add_subplot(gs[2])
        elif i == 3:
            cbar_ax = fig.add_subplot(gs[6])
        elif i == 5:
            cbar_ax = fig.add_subplot(gs[10])
        norm = colors.LogNorm(vmin=v_min, vmax=v_max)

        sns.heatmap(pivoted_data[i],
                    annot=False,
                    cmap="coolwarm",
                    square=True,
                    vmin=v_min, vmax=v_max,  # norm=norm,
                    cbar=(cbar_ax is not None),
                    cbar_ax=cbar_ax,
                    ax=ax)

        # Style cleanup
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_title(configs[i][2], fontweight='bold', family='serif')

        # Label logic: Only on the left-most plot of each pair
        if i == 0:
            ax.set_ylabel("FC Layers", family='serif')
        else:
            ax.set_ylabel("")

        if gs_idx == 9:
            ax.set_xlabel("FE Layers", family='serif')
        else:
            ax.set_xlabel("")

        if cbar_ax:
            cbar_ax.tick_params(labelsize=6, length=0)

    # plt.tight_layout doesn't play well with fixed GridSpec ratios,
    # so we use subplots_adjust for fine-tuning margins
    fig.subplots_adjust(left=0.08, right=0.95, top=0.85, bottom=0.5)
    # plt.tight_layout()

    plt.savefig(output_name, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Uniform plot saved as {output_name}")
    plt.close()


def plot_architecture_matrix_2x3(path_a: str, path_b: str,
                                 output_name: str = "IEEE_Paired_Matrix.pdf"):
    # IEEE 2-column width. Adjusted height for 3 rows.
    fig_width = 3.5  # Increased slightly for better readability in 2-column format
    fig_height = 5

    init_plt()

    # Updated configs: (path, aggregation_method, sub_label/title)
    # We will use the aggregation method name for the left-side labels.
    configs = [
        (path_a, "loop_wise", "Model A"),
        (path_b, "loop_wise", "Model B"),
        (path_a, "majority_vote", ""),
        (path_b, "majority_vote", ""),
        (path_a, "median_loop", ""),
        (path_b, "median_loop", "")
    ]

    # Load and process data
    pivoted_data = []
    for path, agg, label in configs:
        df = pd.read_csv(f"{path}/test_results.csv")
        df['fe_layers'] = df['model_label'].str.split('_').str[2]
        df['fc_layers'] = df['model_label'].str.split('_').str[5]
        df_subset = df[df['aggregation_method'] == agg]

        # FIX: Group by FC then FE, then unstack FE to make FE the COLUMNS (X-axis)
        pivot = df_subset.groupby(['fc_layers', 'fe_layers'])[
            'accuracy'].mean().unstack()
        pivoted_data.append(pivot)

    # Type-wise Scaling (scaling per row/aggregation method)
    v_limits = [
        (min(pivoted_data[0].min().min(), pivoted_data[1].min().min()),
         max(pivoted_data[0].max().max(), pivoted_data[1].max().max())),
        (min(pivoted_data[2].min().min(), pivoted_data[3].min().min()),
         max(pivoted_data[2].max().max(), pivoted_data[3].max().max())),
        (min(pivoted_data[4].min().min(), pivoted_data[5].min().min()),
         max(pivoted_data[4].max().max(), pivoted_data[5].max().max()))
    ]

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(3, 3, width_ratios=[
                          1, 1, 0.05], hspace=0.3, wspace=0.1)

    gs_map = [0, 1, 3, 4, 6, 7]

    for i, gs_idx in enumerate(gs_map):
        ax = fig.add_subplot(gs[gs_idx])
        row_idx = i // 2
        v_min, v_max = v_limits[row_idx]

        # Colorbar logic: One per row, attached to the right-most grid spec
        cbar_ax = None
        if i == 1:
            cbar_ax = fig.add_subplot(gs[2])
        elif i == 3:
            cbar_ax = fig.add_subplot(gs[5])
        elif i == 5:
            cbar_ax = fig.add_subplot(gs[8])

        sns.heatmap(pivoted_data[i],
                    annot=False,
                    cmap="coolwarm",
                    square=True,
                    vmin=v_min, vmax=v_max,
                    cbar=(cbar_ax is not None),
                    cbar_ax=cbar_ax,
                    ax=ax)

        # Title cleanup
        ax.set_title(configs[i][2], fontsize=10, fontweight='bold', family='serif')

        # Y-Label Logic: Aggregation method name on the far left of each row
        if i % 2 == 0:
            agg_name = configs[i][1].replace('_', ' ').title()
            if agg_name == "Majority Vote":
                agg_name = "Soft \ vote"
            elif agg_name == "Median Loop":
                agg_name = "Median \ loop"
            elif agg_name == "Loop Wise":
                agg_name = "Loop-wise"
            ax.set_ylabel(f"$\\bf{{{agg_name}}}$\n\nFC Layers", family='serif')
        else:
            ax.set_ylabel("")
            # Hide labels on the right-side plots for cleanliness
            ax.set_yticklabels([])

        # X-Label Logic: Only on the bottom row
        if i >= 4:
            ax.set_xlabel("FE Layers", family='serif')
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        ax.tick_params(axis='both', which='both', length=0)

        if cbar_ax:
            cbar_ax.tick_params(labelsize=8, length=0)

    plt.savefig(output_name, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Corrected plot saved as {output_name}")
    plt.close()


def plot_architecture_colors(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['fe_layers'] = df['model_label'].str.split('_').str[2]
    df['fc_layers'] = df['model_label'].str.split('_').str[5]

    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['fe_layers', 'fc_layers'])[
            'accuracy'].max().reset_index()
        avg_result = avg_df.pivot(
            index='fc_layers', columns='fe_layers', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    'Feature extractor layers',
                    'Fully connected layers',
                    f"{experiment_path}/max_architecture_matrix_{agg_type}.png")


def plot_lw_matrix(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['lw'] = df['model_label'].str.split('_').str[1]
    df['model'] = df['model_label'].str.split('_').str[3]
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['lw', 'model'])[
            'accuracy'].mean().reset_index()
        avg_result = avg_df.pivot(
            index='model', columns='lw', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    'Linewidth',
                    'Model',
                    f"{experiment_path}/architecture_matrix_{agg_type}.png")


def plot_ch_matrix(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['ch'] = df['model_label'].str.split('_').str[1]
    df['model'] = df['model_label'].str.split('_').str[3]
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['ch', 'model'])[
            'accuracy'].mean().reset_index()
        avg_result = avg_df.pivot(
            index='ch', columns='model', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    'Model',
                    'Channels',
                    f"{experiment_path}/accuracy_matrix_{agg_type}.png")


def plot_lw_matrix_v2(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['lw'] = df['model_label'].str.split('_').str[1]
    df['model'] = df['model_label'].str.split('_').str[3]
    res_options = [f"{i}/128" for i in range(1, 11)]

    def map_resolutions(group):
        unique_lws = sorted(group['lw'].unique())
        mapping = dict(zip(unique_lws, res_options))
        return group['lw'].map(mapping)

    df['resolutions'] = df.groupby(
        'model', group_keys=False).apply(map_resolutions)
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['resolutions', 'model'])[
            'accuracy'].mean().reset_index()
        avg_result = avg_df.pivot(
            index='model', columns='resolutions', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    'Resolutions',
                    'Model',
                    f"{experiment_path}/architecture_matrix_{agg_type}.png")


def plot_lw_plot(experiment_path: str):
    init_plt()

    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['model'] = df['model_label'].str.split('_').str[3]

    df['lw'] = df['model_label'].str.split('_').str[1].astype(int)

    df.sort_values(by=['model', 'lw'], inplace=True)
    df['lw_rank'] = df.groupby('model')['lw'].rank(method='dense').astype(int)
    print(df[df['model'] == "resnet50"][['lw_rank', 'lw']])

    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]

    fig, axes = plt.subplots(3, 1, figsize=(7, 5))

    for agg_idx, agg_type in enumerate(aggregation_types):
        df_subset = df[df['aggregation_method'] == agg_type]
        ax = axes[agg_idx]

        unique_lw = sorted(df_subset['lw_rank'].unique())
        unique_models = df_subset['model'].unique()
        color_map = {model: STYLE_DICT[i % 80]['color']
                     for i, model in enumerate(unique_models)}

        x_pos = 0
        x_ticks = []
        x_labels = []

        for i, model in enumerate(unique_models):
            for lw_rank in df_subset['lw_rank'].unique():
                data = df_subset[(df_subset['lw_rank'] == lw_rank) & (
                    df_subset['model'] == model)]['accuracy'].values
                if len(data) > 0:
                    mean = np.mean(data)
                    std = np.std(data)
                    ax.errorbar(
                        x_pos, mean, yerr=std, fmt='o', color=color_map[model],
                        markersize=8, capsize=5, capthick=2)
                    x_pos += 1
            x_ticks.append(x_pos - len(unique_models) / 2 - 0.5)
            x_labels.append(lw_rank)
            x_pos += 1

        ax.set_xlabel(r'Resolution (\frac{1}{128} pixels)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Aggregation: {agg_type}')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker='o', color='w', markerfacecolor=color_map[m],
                markersize=8) for m in unique_models]
        ax.legend(handles, unique_models, loc='upper right',
                  bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.savefig(f"{experiment_path}/lw_boxplot_combined.pdf")
    plt.close()


def plot_lw_plot2(experiment_path: str):

    init_plt()

    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['model'] = df['model_label'].str.split('_').str[3]
    df['lw'] = df['model_label'].str.split('_').str[1].astype(int)
    df.sort_values(by=['model', 'lw'], inplace=True)
    df['lw_rank'] = df.groupby('model')['lw'].rank(method='dense').astype(int)

    aggregation_types = ["loop_wise",  "majority_vote", "median_loop"]
    unique_models = df['model'].unique()
    color_map = {model: STYLE_DICT[i % 80]['color']
                 for i, model in enumerate(unique_models)}
    dark_color_map = {model: STYLE_DICT_DARK[i % 80]['color']
                      for i, model in enumerate(unique_models)}

    fig, axes = plt.subplots(3, 1, figsize=(PAGE_WIDTH_INCHES, 4), sharex=True,
                             gridspec_kw={'hspace': 0.3})

    for agg_idx, agg_type in enumerate(aggregation_types):
        df_subset = df[df['aggregation_method'] == agg_type]
        ax = axes[agg_idx]

        x_pos = 0
        all_x_ticks = []
        all_x_labels = []

        for model in unique_models:
            model_subset = df_subset[df_subset['model'] == model]

            # --- START EXTENSION: Track range for model mean ---
            start_x = x_pos
            model_accuracies = []

            for lw_rank in sorted(model_subset['lw_rank'].unique()):
                data = model_subset[model_subset['lw_rank']
                                    == lw_rank]['accuracy'].values
                if len(data) > 0:
                    mean = np.mean(data)
                    std = np.std(data)
                    ax.errorbar(x_pos, mean, yerr=std, fmt='o',
                                color=color_map[model], markersize=5,
                                capsize=3, capthick=1.0, zorder=3)

                    model_accuracies.extend(data)
                    all_x_ticks.append(x_pos)
                    all_x_labels.append(lw_rank)
                    x_pos += 1

            if model_accuracies:
                overall_model_mean = np.mean(model_accuracies)
                ax.hlines(
                    overall_model_mean, start_x - 0.3, x_pos - 0.7,
                    colors=dark_color_map[model],
                    linestyles='--', alpha=0.9, zorder=2)

            x_pos += 1  # Gap between models
        agg_name = {
            "loop_wise": "Loop-wise",
            "median_loop": "Median loop",
            "majority_vote": "Soft vote"
        }
        ax.set_title(f'{AGGREGATION_LABELS.get(agg_type, agg_type)}', loc='left',
                     fontweight='bold', pad=5, family='serif')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        ax.set_xticks(all_x_ticks)
        if agg_idx == len(aggregation_types) - 1:
            ax.set_xticklabels([str(x) for x in all_x_labels])
            ax.set_xlabel(
                r'$\mathrm{Resolution \ (\frac{1}{128} \ pixels)}$', family='serif',
                size=11)
        else:
            ax.tick_params(labelbottom=False)

        if agg_idx == 0:
            ax.set_yticks([65, 70, 75])
        elif agg_idx == 1:
            ax.set_yticks([75, 80, 85])
        elif agg_idx == 2:
            ax.set_yticks([70, 75, 80])

    model_nameing_dict = {
        "resnet50": "ResNet-50",
        "efficientnet": "EfficientNet-B1",
        "linearcnn": "Model B",
        "flexicnn": "Model A",
        "efficientnet_b4": "EfficientNet-B4",
        "efficientnet_b5": "EfficientNet-B5",
        "efficientnet_b6": "EfficientNet-B6",
        "efficientnet_b7": "EfficientNet-B7",
    }

    handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker='o', color='w', markerfacecolor=color_map[m],
            markersize=8) for m in unique_models]
    # Add a dummy handle for the mean line

    labels = [model_nameing_dict[m] for m in unique_models]

    fig.legend(handles,
               labels,
               loc='upper center',
               bbox_to_anchor=(0.6, 0.94),
               ncol=len(unique_models),
               frameon=False,
               handletextpad=0.1,
               columnspacing=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_name = f"{experiment_path}/lw_boxplot_v4_with_means.pdf"
    plt.savefig(fig_name, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Combined plot with means saved as {fig_name}")
    plt.close()


def plot_efficient_matrix(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['version'] = df['model_label'].str.split('_').str[2]
    df['eff'] = df['model_label'].str.split('_').str[1]
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['version', 'eff'])[
            'accuracy'].mean().reset_index()
        avg_result = avg_df.pivot(
            index='eff', columns='version', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    'Version',
                    '',
                    f"{experiment_path}/eff_matrix_{agg_type}.png")


def plot_efficient_matrix2(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['version'] = df['model_label'].str.split('_').str[2]
    df['eff'] = df['model_label'].str.split('_').str[1]
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]

    avg_df = df.groupby(['version', 'aggregation_method'])[
        'accuracy'].mean().reset_index()
    avg_result = avg_df.pivot(
        index='version', columns='aggregation_method', values='accuracy')
    plot_matrix(avg_result,
                f'',
                'Aggregation Method',
                'EfficientNet Version',
                f"{experiment_path}/eff_matrix.png")


def plot_resnet_training_matrix():
    df = pd.read_csv(
        f"IB_DL/experiments/default_exp/result/2026-01-23_15-03-16/test_results.csv")
    df['learning_rate'] = df['model_label'].apply(
        lambda x: x.split('rate_')[1].split('_')[0])
    df['optimizer'] = df['model_label'].apply(
        lambda x: x.split('optimizer_')[1].split('_')[0])
    df['pretrained'] = df['model_label'].apply(lambda x: 'pretrained' in x)
    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    df_subset = df
    avg_df = df_subset.groupby(['optimizer', 'pretrained', 'learning_rate'])[
        'accuracy'].mean().reset_index()
    # avg_result = avg_df.pivot(
    #     index='label', columns='model', values='accuracy')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Accuracy Heatmap - Pretrained Models', fontsize=14)

    pretrain_states = [False, True]
    optimizer_states = ["SGD", "Adam"]

    for row, pre_t in enumerate(pretrain_states):
        df_frozen = df_subset[df_subset['pretrained'] == pre_t]
        for col, opt in enumerate(optimizer_states):
            ax = axes[row, col]
            df_group = df_frozen[df_frozen['optimizer'] == opt]

            # Create matrix: rows=learning_rate, cols=aggregation_types
            avg_df = df_group.groupby(['learning_rate', 'aggregation_method'])[
                'accuracy'].mean().reset_index()
            matrix = avg_df.pivot(
                index='learning_rate', columns='aggregation_method', values='accuracy')

            sns.heatmap(matrix, annot=True, fmt=".2f", cbar=False,
                        cmap="coolwarm", ax=ax, square=True, vmin=6, vmax=85)
            ax.set_title(f'{opt} - Pretrained: {pre_t}')
            ax.set_ylabel('Learning Rate' if col == 0 else '')
            ax.set_xlabel('Aggregation' if row == 1 else '')

    plt.tight_layout()
    plt.savefig(
        f"IB_DL/experiments/default_exp/result/resnet50_matrix.png")
    plt.close()


def plot_1D_matrix(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    df['label'] = df['model_label'].str.split('_').str[0]
    df['model'] = df['model_label'].str.split('_').str[2]

    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    for agg_type in aggregation_types:
        df_subset = df[df['aggregation_method'] == agg_type]
        avg_df = df_subset.groupby(['label', 'model'])[
            'accuracy'].mean().reset_index()
        avg_result = avg_df.pivot(
            index='label', columns='model', values='accuracy')
        plot_matrix(avg_result,
                    f'Accuracy Heatmap - Aggregation: {agg_type}',
                    '',
                    '',
                    f"{experiment_path}/architecture_matrix_{agg_type}.png")


def model_labelind(model_name: str):
    model_options = ["resnet50_pretrained", "resnet50",
                     "SIN", "SIN_IMGNET", 'QD_pretrained', 'QD']
    if 'SIN' in model_name:
        return 'SIN_IMGNET' if 'IMGNET' in model_name else 'SIN'
    elif 'QD' in model_name:
        return 'QD_pretrained' if 'pretrained' in model_name else 'QD'
    elif 'resnet50' in model_name:
        return 'resnet50_pretrained' if 'pretrained' in model_name else 'resnet50'
    else:
        return model_name


def plot_pretrain_matrix(experiment_path: str):
    df = pd.read_csv(f"{experiment_path}/test_results.csv")
    model_options = ["resnet50_pretrained", "resnet50",
                     "SIN", "SIN_IMGNET", 'QD_pretrained', 'QD']
    df['model'] = df['model_label'].apply(lambda x: model_labelind(x))
    df['frozen'] = df['model_label'].apply(
        lambda x: True if 'frozen' in x else False)

    aggregation_types = ["loop_wise", "median_loop", "majority_vote"]
    df_subset = df
    avg_df = df_subset.groupby(['model', 'frozen'])[
        'accuracy'].mean().reset_index()
    # avg_result = avg_df.pivot(
    #     index='label', columns='model', values='accuracy')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Accuracy Heatmap - Pretrained Models', fontsize=14)

    frozen_states = [False, True]
    model_groups = [
        ["resnet50", "SIN", 'QD'],
        ["resnet50_pretrained", "SIN_IMGNET", 'QD_pretrained']
    ]

    for row, frozen in enumerate(frozen_states):
        df_frozen = df_subset[df_subset['frozen'] == frozen]
        for col, model_group in enumerate(model_groups):
            ax = axes[row, col]
            df_group = df_frozen[df_frozen['model'].isin(model_group)]

            # Create matrix: rows=models, cols=aggregation_types
            pivot_data = []
            avg_df = df_group.groupby(['aggregation_method', 'model'])[
                'accuracy'].mean().reset_index()
            pivot_data = avg_df.pivot(
                index='model', columns='aggregation_method', values='accuracy')

            matrix = pd.DataFrame(
                pivot_data, index=model_group, columns=aggregation_types)
            sns.heatmap(matrix, annot=True, fmt=".2f", cbar=False,
                        cmap="coolwarm", ax=ax, square=True, vmin=6, vmax=85)
            ax.set_title(f'Frozen: {frozen}')
            sns.heatmap(matrix, annot=True, fmt=".2f", cbar=False,
                        cmap="coolwarm", ax=ax, square=True, vmin=6, vmax=85)
            ax.set_ylabel('Model' if col == 0 else '')
            ax.set_xlabel('Aggregation' if row == 1 else '')

    plt.tight_layout()
    plt.savefig(f"{experiment_path}/pretrain_matrix_.png")
    plt.close()


def rename_dir(directory):

    for filename in os.listdir(directory):
        if filename.endswith(".log"):
            # new_name = "".join([filename[0:9], "_1"+filename[10:0]])
            new_name = filename+"g"
            print(f"Renaming: {filename} -> {new_name}")
            os.rename(os.path.join(directory, filename),
                      os.path.join(directory, new_name))


def visualize_dataset():
    img_size = 256
    lw = 3
    params = {
        'seed': 1, 'img_size': img_size, 'lw': lw, 'experiment_date': 'local_debug',
        'all_measurements': True, 'data_dir': str(PathsConfig.data_dir),
        'num_rows': None, 'test_aggregation': 'all', 'val_aggregation': 'none',
        'persistent_workers': False, 'prefetch_factor': 1, 'num_workers': 1,
        'batch_size': 1, 'class_labels': None, 'ttv_ratio': [0.6, 0.2, 0.2],
        'transformations':
        [{"name": "LoopToTensor"},
         {'name': 'ToImage', 'img_size': img_size, 'linewidth': lw,
          'img_channels': ['f;abs', 'f;rel', 'v;abs']}], }
    _, _, _, ib_dataset = datasets.init_dataset(params)
    samples = []
    labels = []
    for i in range(len(ib_dataset)):
        sample, label = ib_dataset[i]
        if label not in labels:
            labels.append(label)
            samples.append(sample)
        if len(labels) >= 3:
            break

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (sample, label) in enumerate(zip(samples, labels)):
        axes[i].imshow(sample[0], cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig("IB_DL/images/dataset_visualization.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labeels = ['f;abs', 'f;rel', 'v;abs']
    for i in range(3):
        axes[i].imshow(samples[0][i], cmap='gray')
        axes[i].set_title(f'Label: {labeels[i]}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig("IB_DL/images/channel_visualization.png")


def visualize_scales(idx_healthy, idx_copd, idx_ild):
    img_size = 128
    lw = 3
    init_plt()

    # We need all 4 channels for the 4 rows
    channels = ['f;abs', 'f;rel', 'v;abs', 'v;rel']

    # Ensure num_rows handles the maximum index requested
    max_idx = max(idx_healthy, idx_copd, idx_ild)

    params = {
        'seed': 1,
        'img_size': img_size,
        'lw': lw,
        'experiment_date': 'local_debug',
        'all_measurements': True,
        'data_dir': str(PathsConfig.data_dir),
        'num_rows': None,
        'test_aggregation': 'all',
        'val_aggregation': 'none',
        'persistent_workers': False,
        'prefetch_factor': 1,
        'num_workers': 1,
        'batch_size': 1,
        'class_labels': None,
        'ttv_ratio': [0.6, 0.2, 0.2],
        'transformations': [
            {"name": "LoopToTensor"},
            {'name': 'ToImage', 'img_size': img_size,
                'linewidth': lw, 'img_channels': channels}
        ],
    }

    print("Initializing dataset...")
    _, _, _, ib_dataset = datasets.init_dataset(params)
    print("Dataset initialized. Visualizing samples...")

    # Fetch the samples using the provided indices
    # ib_dataset[idx] returns a tuple (sample, label), so we take [0] to get the sample images
    samples = [
        ib_dataset[idx_healthy][0],
        ib_dataset[idx_copd][0],
        ib_dataset[idx_ild][0]
    ]

    col_titles = ['Healthy', 'COPD', 'ILD']
    row_labels = [r'$\mathrm{\dot{V} \ abs.}$', r'$\mathrm{\dot{V} \ rel.}$',
                  r'$\mathrm{V \ abs.}$', r'$\mathrm{V \ rel.}$']

    # Increase figure dimensions to accommodate 4 rows and 3 columns comfortably
    fig_width = 8
    fig_height = 6

    init_plt()

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(
        COLUMN_WIDTH_INCHES*4/3, COLUMN_WIDTH_INCHES))

    for row in range(4):
        for col in range(3):
            ax = axes[col, row]

            # Plot the specific channel (row) for the specific patient (col)
            # Assuming sample shape is (channels, height, width)
            ax.imshow(samples[col][row], cmap='gray_r')

            # Set patient titles on the top row
            if row == 0:
                ax.set_ylabel(col_titles[col], fontsize=12,
                              labelpad=10,  family='serif')

            # Set channel labels on the leftmost column
            if col == 0:
                ax.set_title(row_labels[row], fontsize=12,
                             pad=10,  family='serif')

            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Standardize border lines
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('IB_DL/images/scale_img.pdf', format='pdf', dpi=300)
    print("Plot saved to 'IB_DL/images/scale_img.pdf'")


def sweep_linewidts(resolution, img_size_list):
    """
    Create a grid of images for different line widths and image sizes.

    :param line_width_list: list of line widths to test
    :param img_size_list: list of image sizes to test
    """
    from IB_DL.transformations import build_transformations
    init_plt()

    combination = itertools.product(resolution, img_size_list)
    samples = []
    lws = []
    img_sizes = []
    resolutions = []
    ib_dataset = None
    for res, img_size in combination:
        lw = math.floor(img_size * res)
        params = {'seed': 1, 'img_size': img_size, 'lw': lw,
                  'experiment_date': 'local_debug', 'all_measurements': True,
                  'data_dir': str(PathsConfig.data_dir),
                  'num_rows': None, 'test_aggregation': 'all',
                  'val_aggregation': 'none', 'persistent_workers': False,
                  'prefetch_factor': 1, 'num_workers': 1, 'batch_size': 1,
                  'class_labels': None, 'ttv_ratio': [0.6, 0.2, 0.2],
                  'transformations':
                  [{"name": "LoopToTensor"},
                   {'name': 'ToImage', 'img_size': img_size, 'linewidth': lw,
                    'img_channels': ['f;abs', 'f;rel', 'v;abs']}], }
        if ib_dataset is None:
            _, _, _, ib_dataset = datasets.init_dataset(params)
        else:
            ib_dataset.transform = build_transformations(params)
        sample, label = ib_dataset[10000]
        samples.append(sample)
        lws.append(lw)
        resolutions.append(res)
        img_sizes.append(img_size)

    samples_df = pd.DataFrame(
        {'sample': samples, 'line_width': lws, 'img_size': img_sizes,
         'res': resolutions})

    num_rows = len(img_size_list)
    num_cols = len(resolution)

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * (num_rows / num_cols)),
        constrained_layout=True)

    if num_rows == 1:
        axes = axes[np.newaxis, :]
    if num_cols == 1:
        axes = axes[:, np.newaxis]

    unique_res = sorted(samples_df['res'].unique())
    unique_sizes = sorted(samples_df['img_size'].unique())

    for idx, row in samples_df.iterrows():
        i = unique_sizes.index(row['img_size'])
        j = unique_res.index(row['res'])
        ax = axes[i, j]

        ax.imshow(row['sample'][0], cmap='gray_r')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

        if i == num_rows - 1:
            ax.set_xlabel(f"{int(row['res']*128)}/128 pixels",)

        if j == 0:
            ax.set_ylabel(f"{row['img_size']} pixels")

    fig.text(0.0, 0.5, 'Image size', va='center', ha='center',
             rotation='vertical', fontweight='bold', family='serif')

    fig.text(0.5, 0, 'Resolution', va='center', ha='center',
             fontweight='bold', family='serif')
    plt.tight_layout()
    # fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0, wspace=0)
    fig.savefig("IB_DL/images/img_size-lw/ieee_one_column_grid.pdf",
                format='pdf', dpi=300, bbox_inches='tight')
    print("Grid plot saved as IB_DL/images/img_size-lw/ieee_one_column_grid.pdf")
    plt.close()


def aug_illustration():
    y_title = "$\\mathrm{Zrs \ (hPa \cdot s \cdot L^{-1})}$"
    SUBFIG_TITLES = ["Augmented Loops", "", ""]
    x_label = "$\\mathrm{\dot{V} \ (L \cdot s^{-1})}$"

    img_size = 128
    lw = 3
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
        'transformations':  [
            {"name": "LoopToTensor"},
            {"name": "AWG", "max_mu": 0.1, "sigma": 0.15, "p": 0.5},
            {"name": "SaltAndPepperNoise", "salt_prob": 0.05,
             "pepper_prob": 0.05, "max_amplitude": 1.0, "p": 0.5},
            {"name": "ShiftLoops", "max_shift": 0.1,
             "max_num_of_axes": 2, "p": 0.5},
            {"name": "ToImage", "img_size": img_size, "linewidth": lw,
             "img_channels": ['f;abs', 'f;rel'], "num_skips": 3, "max_len_skips": 8, "change_lwidth": False},
            {"name": "RandomRotation", "degree": 8, "p": 0.5},
            {"name": "GaussianBlur", "kernel_size": (
                3, 3), "sigma": (0.1, 1), "p": 0.5}
        ]}
    _, _, _, ib_dataset = datasets.init_dataset(params)
    sample1 = ib_dataset[0][0]

    params2 = {
        'seed': 1, 'img_size': img_size, 'lw': lw, 'experiment_date': 'local_debug',
        'all_measurements': True, 'data_dir': str(PathsConfig.data_dir),
        'num_rows': 100, 'test_aggregation': 'all', 'val_aggregation': 'none',
        'persistent_workers': False, 'prefetch_factor': 1, 'num_workers': 1,
        'batch_size': 1, 'class_labels': None, 'ttv_ratio': [0.6, 0.2, 0.2],
        'transformations':
        [{"name": "LoopToTensor"},
         {'name': 'ToImage', 'img_size': img_size, 'linewidth': lw,
          'img_channels': ['f;abs', 'f;rel', 'v;abs']}], }
    train_dataloader, _, _, ib_dataset2 = datasets.init_dataset(params2)
    # train_features, train_labels = next(iter(train_dataloader))

    fig = plt.figure(figsize=(12, 6))

    gs = fig.add_gridspec(3, 4, width_ratios=[
                          1.2, 3, 1.2, 3], wspace=0.02, hspace=0.0)

    colors = ["#BBF5EB", "#FDFDFD",]

    for row in range(2):
        for col in range(4):
            bg_color = colors[0] if col % 2 == 0 else colors[1]
            subfig = fig.add_subfigure(gs[row, col])
            subfig.set_facecolor(bg_color)

            if col % 2 == 1:
                subfig.subplots_adjust(
                    left=0.0, right=1, bottom=0.00, top=0.82, wspace=0.1)

                axs = subfig.subplots(1, 3, sharex=True, sharey=True)
                subfig.suptitle(
                    SUBFIG_TITLES[row],
                    x=0.12, y=0.92, fontsize=11, fontweight='bold', ha='left')

                for i, ax in enumerate(axs):
                    if col == 1:
                        np.random.seed(random.randint(0, 10000))
                        ax.imshow(ib_dataset[0][0][row], cmap='gray_r')
                        # if i == 2:
                        # ax.set_xlabel(x_label, fontweight='bold', labelpad=5)
                    if col == 3:
                        ax.imshow(ib_dataset[1][0][row], cmap='gray_r')
                        # if i == 2:
                        # ax.set_xlabel(x_label, fontweight='bold', labelpad=5)
                    ax.set_facecolor('white')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # if i == 0:
                    #     ax.set_ylabel(y_title, fontweight='bold', labelpad=2)

            else:
                subfig.subplots_adjust(
                    left=0.2, right=0.8, bottom=0.2, top=0.82)

                ax = subfig.subplots(1, 1)
                if row == 0:
                    subfig.suptitle('Original loop', x=0.5, y=0.92,
                                    fontsize=11, fontweight='bold')
                if col == 0:
                    ax.imshow(ib_dataset2[0][0][row], cmap='gray_r')
                if col == 2:
                    ax.imshow(ib_dataset2[1][0][row], cmap='gray_r')
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_facecolor('white')

    plt.savefig('IB_DL/images/augmentation_illustration.pdf',
                format='pdf', dpi=300)


def augmentation_illustration():
    from IB_DL.transformations import build_transformations

    y_title = "$\\mathrm{Zrs \ (hPa \cdot s \cdot L^{-1})}$"
    SUBFIG_TITLES = ["Augmented Loops", "", ""]
    x_label = "$\\mathrm{\dot{V} \ (L \cdot s^{-1})}$"

    img_size = 128
    lw = 3

    transformations = [
        {"name": "LoopToTensor"},
        {"name": "AWG", "max_mu": 0.1, "sigma": 0.15, "p": 1},
        {"name": "SaltAndPepperNoise", "salt_prob": 0.05, "pepper_prob": 0.05,
         "max_amplitude": 1.0, "p": 1},
        {"name": "ShiftLoops", "max_shift": 0.1, "max_num_of_axes": 2, "p": 1},
        {"name": "ToImage", "img_size": img_size, "linewidth": lw,
         "img_channels": ['f;abs', 'f;rel'],
         "num_skips": 3, "max_len_skips": 8, "change_lwidth": False},
        {"name": "ToImage", "img_size": img_size, "linewidth": lw,
         "img_channels": ['f;abs', 'f;rel'],
         "num_skips": 0, "max_len_skips": 8, "change_lwidth": False},
        {"name": "RandomRotation", "degree": 10, "p": 1},
        {"name": "GaussianBlur", "kernel_size": (3, 3),
         "sigma": (0.5, 1),
         "p": 1}]
    transform_combos = [
        [0, 5],
        [0, 1, 5],
        [0, 2, 5],
        [0, 3, 5],
        [0, 4],
        [0, 5, 6],
        [0, 5, 7]
    ]
    names = ["Original loop", "GWN", "Salt & Pepper",
             "Shift",  "Dropout", "Rotation", "Gaussian blur"]

    params2 = {
        'seed': 1, 'img_size': img_size, 'lw': lw, 'experiment_date': 'local_debug',
        'all_measurements': True, 'data_dir': str(PathsConfig.data_dir),
        'num_rows': 100, 'test_aggregation': 'all', 'val_aggregation': 'none',
        'persistent_workers': False, 'prefetch_factor': 1, 'num_workers': 1,
        'batch_size': 1, 'class_labels': None, 'ttv_ratio': [0.6, 0.2, 0.2],
        'transformations':
        [{"name": "LoopToTensor"},
         {'name': 'ToImage', 'img_size': img_size, 'linewidth': lw,
          'img_channels': ['f;abs', 'f;rel', 'v;abs']}], }
    _, _, _, ib_dataset2 = datasets.init_dataset(params2)
    init_plt()

    # Slightly increased height to accommodate titles cleanly
    fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, 2.5))

    # Increased wspace and hspace to give the images and titles breathing room
    gs = fig.add_gridspec(2, 7, wspace=0.05, hspace=0.0)

    for row in range(2):
        for col in range(7):
            params2["transformations"] = [transformations[i]
                                          for i in transform_combos[col]]
            ib_dataset2.transform = build_transformations(params2)

            ax = fig.add_subplot(gs[row, col])

            # FIX 1: Only add titles to the top row!
            # Removed the hacky x/y coordinates and used 'pad' to push it up slightly.
            if row == 0:
                ax.set_title(names[col], fontsize=10,
                             fontweight='bold', pad=6, ha='center')

            np.random.seed(random.randint(0, 10000))
            sample = ib_dataset2[10][0]

            # The if/else was doing the same thing, simplified it:
            ax.imshow(sample[row], cmap='gray_r')
            ax.set_facecolor('white')

            # Strip ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Make sure borders are drawn cleanly around each image
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)

    # FIX 2: Manually adjust margins to stretch the grid across the figure canvas
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.05)

    # FIX 3: Add bbox_inches='tight' so the saved PDF crops out the massive white borders
    plt.savefig('IB_DL/images/augmentation_illustration.pdf',
                format='pdf', dpi=300, bbox_inches='tight')


def plot_augmentation_strength_vs_prob():
    from IB_DL.transformations import build_transformations
    # Assuming 'datasets' and 'PathsConfig' are imported in your actual file

    img_size = 128
    lw = 3

    # --- 1. Dataset Initialization ---
    params = {
        'seed': 1,
        'img_size': img_size,
        'lw': lw,
        'experiment_date': 'local_debug',
        'all_measurements': True,
        'data_dir': str(PathsConfig.data_dir),
        'num_rows': None,
        'test_aggregation': 'all',
        'val_aggregation': 'none',
        'persistent_workers': False,
        'prefetch_factor': 1,
        'num_workers': 1,
        'batch_size': 1,
        'class_labels': None,
        'ttv_ratio': [0.6, 0.2, 0.2],
        # Will be dynamically updated
        'transformations': [{"name": "LoopToTensor"}]
    }
    _, _, _, ib_dataset = datasets.init_dataset(params)
    init_plt()

    # --- 2. Augmentation Parameters (from your experiment) ---
    num_columns = 5

    # We select 3 levels of strength (indices 0, 2, 4 representing levels 1, 3, 5)
    strength_indices = [0, 2, 4]
    # We select 3 levels of probability (Low, Med, High)
    p_selected = [0.05, 0.5, 1.0]

    max_mu = np.linspace(0.05, 0.5, num=num_columns)
    sigma = np.linspace(0.1, 1, num=num_columns)
    salt_pepper_prob = np.linspace(0.05, 0.2, num=num_columns)
    max_stretch = np.linspace(0.05, 0.3, num=num_columns)
    max_shift = np.linspace(0.05, 0.3, num=num_columns)
    num_skips = np.array([1, 2, 2, 3, 4])
    max_len_skips = np.linspace(2, 10, num=num_columns, dtype=int)
    random_rot_degree = np.linspace(1, 10, num=num_columns)
    sigma_min_blur = np.linspace(0.05, 0.2, num=num_columns)
    sigma_delta_blur = np.linspace(0.1, 0.5, num=num_columns)

    # --- 3. Figure Setup ---
    # Using a square figure for a 3x3 grid
    fig = plt.figure(figsize=(3, 3))
    gs = fig.add_gridspec(3, 3, wspace=0.05, hspace=0.05)

    # Select a fixed loop index so we compare the exact same base image
    loop_index = ib_dataset.train_indices[-10]

    for row in range(3):
        for col in range(3):
            p_ = p_selected[col]
            i = strength_indices[row]

            # --- 4. Build transformation pipeline for this cell ---
            current_transforms = [
                {"name": "LoopToTensor"},
                {"name": "AWG", "max_mu": max_mu[i],
                    "sigma": sigma[i], "p": p_},
                {"name": "SaltAndPepperNoise",
                    "salt_prob": salt_pepper_prob[i], "pepper_prob": salt_pepper_prob[i], "max_amplitude": 1.0, "p": p_},
                {"name": "StretchLoops",
                    "max_delta_r": max_stretch[i], "max_num_of_axes": 2, "p": p_},
                {"name": "ShiftLoops",
                    "max_shift": max_shift[i], "max_num_of_axes": 2, "p": p_},
                # ToImage splits pre-render (signals) and post-render (pixels)
                {"name": "ToImage", "img_size": img_size, "linewidth": lw,
                 # Using 1 channel for clean visualization
                 "img_channels": ['f;abs'],
                 "num_skips": num_skips[i], "max_len_skips": max_len_skips[i], "change_lwidth": False},
                {"name": "RandomRotation",
                    "degree": random_rot_degree[i], "p": p_},
                {"name": "GaussianBlur", "kernel_size": (3, 3), "sigma": (
                    sigma_min_blur[i], sigma_min_blur[i] + sigma_delta_blur[i]), "p": p_}
            ]

            params["transformations"] = current_transforms
            ib_dataset.transform = build_transformations(params)

            # Get the same loop but apply random transforms
            sample = ib_dataset[loop_index][0]

            # --- 5. Plotting ---
            ax = fig.add_subplot(gs[row, col])

            # Plotting channel 0 (f;abs)
            ax.imshow(sample[0], cmap='gray_r')
            ax.set_facecolor('white')
            ax.set_xticks([])
            ax.set_yticks([])

            # Borders
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)

            # --- 6. Labeling ---
            # Top row gets Probability titles
            if row == 0:
                ax.set_title(f"p = {p_}", fontsize=11,
                             pad=8)

            # Left column gets Strength labels
            if col == 0:
                # Level 1, 3, 5 corresponding to indices 0, 2, 4
                ax.set_ylabel(
                    f"Strength {i+1}", fontsize=11,  labelpad=8)

    # Adjust margins to fit titles/labels
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.02)

    output_path = 'IB_DL/images/augmentation_3x3_grid.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved 3x3 augmentation grid to {output_path}")
    plt.close()


def combined_experiment_visualization(directory, agg='mean'):
    df = pd.read_csv(f"{directory}/test_results.csv")

    # --- 1. Extraction Logic ---
    df['model'] = df['model_label'].apply(
        lambda x: x.split('_optimizer')[0].split('model_')[1])
    df['optimizer'] = df['model_label'].apply(
        lambda x: x.split('optimizer_')[1].split('_tf')[0])
    df['tf_strength'] = df['model_label'].apply(
        lambda x: x.split('trength_')[1].split('_img-channels')[0])
    df['img_channels'] = df['model_label'].apply(
        lambda x: x.split('img-channels_')[1].split('_seed')[0])

    # Assume calculate_f1 is defined globally
    df['f1_score'] = df['confusion_matrix_str'].apply(calculate_f1)

    # Check if F1 is 0-1 scale and convert to 0-100 percentage
    if df['f1_score'].max() <= 1.05:
        df['f1_score'] = df['f1_score'] * 100

    group_cols = ['model', 'optimizer', 'tf_strength',
                  'img_channels', 'aggregation_method']

    # --- 2. Aggregate Data ---
    df_grouped = df.groupby(group_cols).agg({
        'accuracy': agg,
        'f1_score': agg
    }).reset_index()
    df_grouped = df_grouped[df_grouped['accuracy'] > 60]

    df_max = df.groupby(group_cols).agg({
        'accuracy': 'max',
        'f1_score': 'max'
    }).reset_index()
    df_max = df_max[df_max['accuracy'] > 60]

    # --- 3. Setup Subplots (3 Rows, 2 Columns) ---
    # sharex and sharey automatically clean up redundant inner tick labels
    fig, axes = plt.subplots(3, 2, figsize=(
        10, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    plot_configs = [
        {'data': df_grouped, 'hue': 'model', 'title': 'By Model Architecture'},
        {'data': df_grouped, 'hue': 'optimizer', 'title': 'By Optimizer'},
        {'data': df_grouped, 'hue': 'tf_strength',
            'title': 'By Transform Strength'},
        {'data': df_grouped, 'hue': 'img_channels', 'title': 'By Image Channels'},
        {'data': df_grouped, 'hue': 'aggregation_method',
            'title': f'By Aggregation Method ({agg.capitalize()})'},
        {'data': df_max, 'hue': 'model',
            'title': 'By Model Architecture (Max for seeds)'}
    ]

    # Set hard limits for 60% to 90% as requested
    min_val, max_val = 75, 83
    # --- 4. Plotting Loop ---
    for i, conf in enumerate(plot_configs):
        ax = axes[i]

        sns.scatterplot(
            data=conf['data'],
            x='f1_score',
            y='accuracy',
            hue=conf['hue'],
            ax=ax,
            s=70,
            alpha=0.8,
            edgecolor='w',
            linewidth=0.5
        )
        if i == 4:
            min_val, max_val = 62, 90
        elif i == 5:
            min_val, max_val = 78, 88
        else:
            min_val, max_val = 75, 83

        # Draw the y=x dashed diagonal line matching the new 60-90 scale
        ax.plot([min_val, max_val], [min_val, max_val],
                ls="--", c="gray", alpha=0.5, zorder=0)

        # Styling & Titles
        ax.set_title(conf['title'], fontweight='bold', pad=12, fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Enforce uniform square limits 60-90
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])

        # Smart label placement: only show X labels on the bottom row, Y on the left col
        if i >= 4:
            ax.set_xlabel("Macro F1 Score (%)", fontweight='bold')
        else:
            ax.set_xlabel("")

        if i % 2 == 0:
            ax.set_ylabel("Accuracy (%)", fontweight='bold')
        else:
            ax.set_ylabel("")

        # Customize Legend
        handles, labels = ax.get_legend_handles_labels()

        # Strip out the seaborn hue column name header
        if labels and labels[0] == conf['hue']:
            handles, labels = handles[1:], labels[1:]

        # Move legend inside to the upper left (which is usually empty space in an Acc vs F1 plot)
        ax.legend(handles, labels, loc='upper left', frameon=True,
                  edgecolor='lightgray', fontsize=9, title=None)

    # Clean layout and reduce empty space
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.15, wspace=0.05)

    output_file = f"{directory}/combined_scatter_matrix_{agg}.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Saved 3x2 combined visualization to {output_file}")
    plt.close()
