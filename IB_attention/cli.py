from datetime import date, datetime
import os
from typing import Optional
import click
from sympy import S
from IB_attention.train import train
from IB_attention.eval import evaluate, init_logreaders, load_model, calculate_detailed_csv
from IB_attention.datasets import IBDataset, init_dataset
import numpy as np
from IB_attention.config import PathsConfig
from torchvision.transforms import v2  # type: ignore
import IB_attention.transformations as transformations
import torch
from pathlib import Path
import importlib
import inspect
import IB_attention.experiments.batch_size_exp.config as batch_exp
import IB_attention.experiments.linewidth_exp.config as lw_exp
import IB_attention.experiments.pixel_exp.config as px_exp
import IB_attention.utils as utils
import IB_attention.experiments.worker_exp.config as worker_exp
import IB_attention.experiments.default_exp.config as default_exp_config
import IB_attention.experiments.default_config as default_config
import sys
import shlex


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option()
def cli():
    pass


@click.option('--data-path', default=None, help='Path to the training data')
@cli.command()
def data_load_example(data_path: Optional[str | Path]):
    """Example of loading dataset.
    """
    cfg = PathsConfig()
    if data_path is None:
        data_path = cfg.data_dir
    click.echo("MEDIAN transformations used:")
    ib_dataset = IBDataset([os.path.join(data_path, i) for i in [
        "all_measurements_1.csv", "all_measurements_2.csv", "all_measurements_3.csv"]],
        transform=v2.Compose([transformations.LoopToTensor()]),
        target_transform=None, num_of_rows=200, loop_set=True)
    click.echo(f"Number of pateints: {len(ib_dataset)}")
    click.echo(f"Classes in the dataset: {np.unique(ib_dataset.annotations)}")
    ib_dataset.plot_histogram()

    item, label = ib_dataset[0]
    click.echo(f"First item: {item}")
    click.echo(f"Label of the first item: {label}")
    click.echo(f"Type of the data: {type(item)}")
    click.echo(f"Shape of the data: {item.shape}")  # type: ignore


@click.option('--batch-size', default=32, help='Batch size for training')
@click.option('--tensor', default=False,
              help='Use tensor transformations if True, otherwise use image transformations')
@click.option('--layers', default=True,
              help='If True, use layers for image transformations to visualize')
@click.option('--seed', default=42, help='Seed for reproducibility')
@cli.command()
def visualize_batch(
        batch_size: int, tensor: bool = False, layers: bool = True, seed: int = 42):
    """
    Example of loading dataset and visualizing a batch.
    """
    params = {
        "batch_size": batch_size,
        "data_dir": str(PathsConfig.data_dir),
        "ttv_ratio":  [0.7, 0.2, 0.1],
        "transformations": [
            {"name": "MedianLoops"},
            {"name": "LoopToTensor"},
            {"name": "Flatten"}
        ] if tensor else [
            {"name": "LoopToTensor"},
            {"name": "ToImage", "img_size": 28, "linewidth": 1, "img_channels": [
                "f;abs", "f;rel", "f;zc", "v;abs", "v;rel", "v;zc"], "num_skips": 0, "max_len_skips": 0}],
        "num_rows": 500,
        "class_labels": None,
        "all_measurements": True,
        "seed": seed,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "test_aggregation": "none",
    }
    utils.set_seed(seed)
    train_dataloader, validation_dataloader, test_dataloader, ib_dataset = init_dataset(
        params)
    params["class_labels"] = ib_dataset.class_labels
    params["num_classes"] = len(params["class_labels"])
    train_features, train_labels = next(iter(train_dataloader))

    if not tensor:
        if layers:
            utils.visualize_image_data_per_channel(
                train_features, train_labels)
        else:
            utils.visualize_image_data(train_features, train_labels)
    else:
        utils.visualize_tensor_data(
            train_features, train_labels, flattened=True)


@cli.command()
@click.option('--name', default="batch_size_exp", help='Experiment name to run')
def run_experiment(name: str):
    try:
        exp_module = importlib.import_module(
            f"IB_attention.experiments.{name}.config")
        exp_module.run_experiment()
    except ModuleNotFoundError:
        click.echo(f"Experiment '{name}' not found.")


@click.option(
    '--epochs', multiple=True, type=int, default=[2],
    help='Number of epochs (can specify multiple times for a list)')
@click.option(
    '--model', multiple=True, default=["resnet18_pretrained"],
    help='Model(s) to use for training (can specify multiple times for a list)')
@click.option(
    '--num_workers', multiple=True, type=int, default=[0],
    help="Number(s) of workers for data loading (can specify multiple times for a list)")
@click.option(
    '--persistent_workers', multiple=True, type=bool, default=[False],
    help="Use persistent workers for data loading (can specify multiple times for a list)")
@click.option(
    '--prefetch_factor', multiple=True, type=int, default=[None],
    help="Prefetch factor(s) for data loading (can specify multiple times for a list)")
@click.option('--all-measurements', is_flag=True, default=True,
              help="Use all measurements (default: True)")
@click.option('--num-rows', type=int, default=None,
              help="Number of rows to use from the dataset (default: None)")
@cli.command()
def num_worker_experiment(
        epochs, model, num_workers, persistent_workers, prefetch_factor,
        all_measurements, num_rows):
    def unpack(val):
        return val[0] if len(val) == 1 else list(val)
    inpt_dict = {
        "epochs": unpack(epochs),
        "model": unpack(model),
        "num_workers": unpack(num_workers),
        "persistent_workers": unpack(persistent_workers),
        "prefetch_factor": unpack(prefetch_factor),
        "all_measurements": all_measurements,
        "num_rows": num_rows
    }
    inpt_dict['experiment_date'] = str(
        date.today().isoformat()) + "_" + str(datetime.now().strftime('%H-%M-%S'))
    worker_exp.run_experiment(inpt_params=inpt_dict)
    log_dir = os.path.join(os.path.dirname(
        worker_exp.__file__), inpt_dict['experiment_date'])
    log_names = [os.path.join(log_dir, f)
                 for f in os.listdir(log_dir) if f.endswith('.log')]
    evaluate(log_names)  # type: ignore


def get_default_config_params():
    # Instantiate DefaultParams and extract all instance variables
    params = {}
    default_params = default_config.DefaultParams()
    for name, value in vars(default_params).items():
        # Only include simple types (not lists/dicts/None) for CLI options
        if isinstance(value, (str, int, float, bool, type(None))):
            params[name] = value
    return params


def add_dynamic_options(command):
    """
    Decorator to add click options dynamically from default_config.
    """
    params = get_default_config_params()
    for param, default in reversed(list(params.items())):
        if isinstance(default, bool):
            option_type = bool
        elif isinstance(default, int):
            option_type = int
        elif isinstance(default, float):
            option_type = float
        else:
            option_type = str
        command = click.option(
            f'--{param.replace("_", "-")}',
            multiple=True,
            default=[default],
            show_default=True,
            help=f'Set {param} (can specify multiple times for NAS)',
            type=option_type
        )(command)
    return command


@cli.command()
@add_dynamic_options
@click.option('--name', default="default_exp", help='Experiment name to run')
@click.option('--exp_date', type=str, default="",
              help='Experiment ddate to use for continuing an interrupted experiment (format: YYYY-MM-DD_HH-MM-SS)')
@click.option('--eval', 'eval_mode', is_flag=True, default=True,
              help='Run experiment in evaluation mode. default: False')
@click.option('--log_info', is_flag=True, default=True,
              help='Do the logging of info level messages. default: False')
def default_experiment(name, exp_date, eval_mode, log_info, **kwargs):
    """
    Run experiment with parameters from default_config, allowing grid search via CLI.
    """
    # log_command_to_history() # Uncomment this line if you want to log the command history for default_experiment as well
    if not log_info:
        utils.LOG_LEVEL = "NOTHING"

    def unpack(val):
        return val[0] if len(val) == 1 else list(val)
    # Remove None values and unpack
    inpt_dict = {k: unpack(v) for k, v in kwargs.items() if v is not None}
    # Remove values that are the same as the default
    default_params = get_default_config_params()
    filtered_inpt_dict = {k: v for k, v in inpt_dict.items()
                          if k in default_params and v != default_params[k]}
    # Pass eval_mode to the experiment config if needed
    try:
        if exp_date == "":
            filtered_inpt_dict['experiment_date'] = str(
                date.today().isoformat()) + "_" + str(datetime.now().strftime('%H-%M-%S'))

        else:
            filtered_inpt_dict['experiment_date'] = exp_date
        exp_module = importlib.import_module(
            f"IB_attention.experiments.{name}.config")
        log_names = exp_module.run_experiment(inpt_params=filtered_inpt_dict)
        if eval_mode:
            click.echo("Running evaluation on the experiment results...")
            evaluate(log_names)
    except ModuleNotFoundError:
        click.echo(f"Experiment '{name}' not found.")


@cli.command()
@click.option('--name', type=str, default="", help='Experiment name to evalaute')
@click.option('--dir', type=str, default="",
              help='Directory containing experiments name to evalaute')
@click.option('--output_file', type=str, default="",
              help="Number of the latest experiments to evaluate")
# @click.option('--date', multiple=True, type=str,  default="",  help='Date of experiment to evaluate (can specify multiple times for a list)')
def evaluate_log(name, dir, output_file):
    if name:
        print(f"Evaluating experiments for {name}")
        evaluate([name], output_log=output_file)
    if dir:
        print(f"Evaluating all experiments in directory {dir}")
        log_names = [os.path.join(dir, f) for f in sorted(os.listdir(
            dir)) if f.endswith('.log')]
        evaluate(log_names, output_log=output_file)


@cli.command()
@click.option('--name', default="default_exp", help='Experiment name to run')
@click.option('--log_path',  default=None, help='Path of the log file')
@click.option('--eval', 'eval_mode', is_flag=True, default=True,
              help='Run experiment in evaluation mode. default: False')
@click.option('--log_info', is_flag=True, default=True,
              help='Do the logging of info level messages. default: False')
def continue_train(name, log_path, eval_mode, log_info):
    """
    Run experiment with parameters from default_config, allowing grid search via CLI.
    """
    log_command_to_history()
    if not log_info:
        utils.LOG_LEVEL = "NOTHING"
    if log_path is None:
        raise ValueError("Log path must be provided for continue_train.")
    # Pass eval_mode to the experiment config if needed
    try:
        logreaders = init_logreaders(log_files=log_path)
        reader = logreaders[0]

        exp_module = importlib.import_module(
            f"IB_attention.experiments.{name}.config")
        log_names = exp_module.run_experiment(
            inpt_params=reader.params, resume=True)
        if eval_mode:
            click.echo("Running evaluation on the experiment results...")
            evaluate(log_names)
    except ModuleNotFoundError:
        click.echo(f"Experiment '{name}' not found.")


def log_command_to_history():
    full_command = shlex.join(sys.argv)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {full_command}\n"
    with open("command_history.txt", "a") as f:
        f.write(log_entry)


@cli.command()
@click.option('--in-file', default=None, help='Path of the input log file')
@click.option('--out-file',  default=None, help='Path of the output file')
@click.option('--agg', is_flag=True, default=False,
              help='Aggregate results by seed (default: False)')
def detailed_eval_csv(in_file, out_file, agg):
    """
    Create a detailed CSV file from the log file of an experiment, with an option to aggregate results by seed.
    """
    calculate_detailed_csv(in_file, out_file, aggregate_by_seed=agg)
