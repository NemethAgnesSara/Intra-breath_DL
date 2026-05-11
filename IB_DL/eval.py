import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from IB_DL.datasets import IBDataset, init_dataset
from IB_DL.logger import LogReader
import os
import matplotlib.pyplot as plt
from IB_DL import visualize
from typing import Any, Optional
from IB_DL import utils
from IB_DL.models import get_model
from torch.utils.data import DataLoader  # type: ignore
from pathlib import Path
from IB_DL.utils import cprint, init_logreaders
from IB_DL.transformations import MedianLoops
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
from contextlib import redirect_stdout
import traceback
import ast
import re


def evaluate(log_files: list[str | Path] | str | Path,
             labels: Optional[list[str]] = None,
             output_log: Optional[str] = None,
             ib_dataset=None):

    def _run_eval():
        print(f"Evaluating log files-----------------------")
        logreaders = init_logreaders(log_files=log_files)
        test_results_path = os.path.join(
            logreaders[0].params['output_dir'], 'test_results.csv')
        if os.path.exists(test_results_path):
            print(f"test_results.csv found at {test_results_path}")
            acc_df = pd.read_csv(test_results_path)

            acc_df['confusion_matrix'] = acc_df['confusion_matrix_str'].apply(
                lambda x: eval(x))
            new_readrs = []
            for logreader in logreaders:
                try:
                    logreader.params['prefix']
                except Exception as e:
                    print(f"Error processing log reader {logreader.path}: {e}")
                    continue
                if logreader.params['prefix'] in acc_df['model_label'].values:
                    utils.cprint(
                        f"Experiment: {logreader.params['prefix']} already evaluated. Skipping.",
                        "yellow")
                else:
                    #     logreader.params['transfourrmations'] = [
                    #         {"name": "LoopToIBFeates", "feature_selector": "MEAN", "measurement": False}]
                    new_readrs.append(logreader)
            if len(new_readrs) > 0:
                new_acc_df = eval_on_test_smart(logreaders=new_readrs)
                acc_df = pd.concat([acc_df, new_acc_df], ignore_index=True)
                acc_df.to_csv(test_results_path, index=False)

        else:
            print(f"test_results.csv not found at {test_results_path}")
            acc_df = eval_on_test_smart(
                logreaders=logreaders, ib_dataset_param=ib_dataset)

        eval_labels = labels
        if eval_labels is None:
            eval_labels = [reader.params['prefix']
                           for reader in logreaders]

        # acc_df = eval_on_test_smart(logreaders=logreaders)

        # all_acc, all_cm = eval_on_test(logreaders=logreaders)
        visualize.plot_test_accuracies(
            acc_df['accuracy'],
            acc_df['aggregation_method'],
            acc_df['model_label'],
            logreaders[0].params['output_dir'] + "/test_accuracies.pdf")

        for output_item in logreaders[0].params['output_items']:
            if output_item == "loss_plot":
                plot_train_val_values(
                    logreaders=logreaders, value_str="loss", labels=eval_labels,
                    save_path=str(logreaders[0].params['output_dir']) + "/loss_plot.pdf")
            if output_item == "acc_plot":
                plot_train_val_values(
                    logreaders=logreaders, value_str="acc", labels=eval_labels,
                    save_path=str(logreaders[0].params['output_dir']) + "/acc_plot.pdf")
            # if output_item == "conf_matrix":
            #     for i in range(len(logreaders)):
            #         reader = logreaders[i]
            #         cm = acc_df['confusion_matrix'][i]
            #         visualize.plot_conf_matrix(conf_matrix=cm,
            #                                    class_labels=reader.params['class_labels'] if reader.params['class_labels'] is not None else [
            #                                        "healthy", "COPD", "ILD'],
            #                                    save_path=reader.params['output_dir']+"/"+reader.params['prefix']+"_cm.pdf")

        # TODO cams, conf matrix and test eval and also call from cli

    if output_log is not None and output_log.endswith('.txt'):
        print(f"--- Output is being redirected to {output_log} ---")
        with open(output_log, 'w') as f:
            with redirect_stdout(f):
                _run_eval()
    else:
        _run_eval()


def check_accuracy(loader: DataLoader, model: nn.Module, device: str,
                   class_labels: Optional[list[str]] = None, conf_matrix: bool = False
                   ) -> tuple[float, tuple[float, np.ndarray]]:
    """
    Checks the accuracy of the model on the given dataset loader.
    """
    num_correct = 0
    num_samples = len(loader.dataset)  # type: ignore
    all_predictions = []
    all_targets = []

    model.eval()

    class_names = class_labels if class_labels is not None else [
        "healthy", "COPD", "ILD"]
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Test size: {len(loader.dataset)}")  # type: ignore

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            scores = model(x)
            _, predictions = scores.max(1)

            if isinstance(y[0], str):  # If labels are strings
                target = torch.tensor([label_to_idx[y_]
                                      for y_ in y], device=scores.device)
            else:  # Already integer labels
                target = y.to(device)
            num_correct += (predictions == target).sum().item()

            if conf_matrix:
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    accuracy = float(num_correct) / float(num_samples) * 100
    print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
    model.train()

    if conf_matrix:
        if class_labels is None and len(np.unique(all_targets)) > len(class_names):
            print("Warning: More unique target classes found than default class_labels. Confusion matrix may be incorrect.")
            max_label = max(max(all_targets), max(
                all_predictions)) if all_targets else 0
            if max_label >= len(class_names):
                class_names.extend(
                    [f"class_{i}" for i in range(len(class_names), max_label + 1)])

        cm = confusion_matrix(all_targets, all_predictions,
                              labels=list(range(len(class_names))))
        # print("Confusion Matrix:")
        # print(cm)
        return accuracy, cm

    return accuracy


def eval_on_test(logreaders: list[LogReader]):
    """
    Evaluate the model on the test dataset for each log reader.
    The process includes loading the dataset, loading the model,
    checking accuracy, and optionally plotting the confusion matrix.
    """
    all_acc, all_cm = [], []
    for reader in logreaders:
        dataloader = load_dataset(reader, "test")
        model = load_model(reader)
        acc, cm = check_accuracy(loader=dataloader,
                                 model=model,
                                 device=reader.params['device'],
                                 class_labels=reader.params['class_labels'],
                                 conf_matrix=True)  # type: ignore
        all_acc.append(acc)
        all_cm.append(cm)
        utils.cprint(
            f"The accuracy of experiment: {reader.params['prefix']} is: {acc}",
            "magenta")
        # visualize.plot_conf_matrix(conf_matrix=cm,
        #                            class_labels=reader.params['class_labels'] if reader.params['class_labels'] is not None else [
        #                                "healthy", "COPD", "ILD'],
        #                            save_path=reader.params['output_dir']+"/"+reader.params['prefix']+"_cm.pdf")
    return all_acc, all_cm


def get_probabilities(model_output):
    """
    Safely converts model output to probabilities regardless of 
    whether the model outputs logits or pre-softmaxed values.
    """
    # Case 1: Binary Classification (Output shape is [Batch, 1])
    if model_output.shape[1] == 1:
        probs_positive = torch.sigmoid(model_output)
        return torch.cat([1 - probs_positive, probs_positive], dim=1)

    # Case 2: Check if already probabilities (Sum is approx 1 and all positive)
    is_probability = (model_output.min() >= 0) and \
                     (model_output.max() <= 1) and \
        torch.isclose(model_output.sum(dim=1).mean(),
                      torch.tensor(1.0), atol=1e-2)

    if is_probability:
        return model_output

    # Case 3: Standard Logits (Values are unbounded, sum != 1)
    else:
        return F.softmax(model_output, dim=1)


def eval_on_test_smart(
        logreaders: list[LogReader],
        ib_dataset_param: Optional[DataLoader] = None):
    """
    Evaluate the model on the test dataset using different aggregation strategies.
    The process includes loading the dataset, loading the model,
    checking accuracy with various test aggregation methods.
    Results are logged to CSV after all evaluations.
    """
    results_data = []
    output_dir = logreaders[0].params['output_dir']

    # check for already existing results file
    output_df = pd.DataFrame(
        # , "early_stop", "overfitting'])
        columns=['exp_name', 'exp_mode', 'accuracy', 'date', 'max_test_acc', 'max_val_acc', 'epoch'])

    if os.path.exists(os.path.join(output_dir, "out.csv")):
        output_df = pd.read_csv(os.path.join(output_dir, "out.csv"))

    train_epochs, train_acc, _, val_acc = get_train_val_values(
        logreaders, "acc")

    results_data = []

    try:
        for i, reader in enumerate(logreaders):
            for transformation in reader.params['transformations'].copy():
                if transformation['name'] == "ToMeasurements":
                    reader.params['transformations'] = [
                        {"name": "LoopToIBFeatures", "feature_selector": "MEAN", "measurement": False}]
            try:
                if (output_df['exp_name'] == reader.params['prefix']).any():
                    utils.cprint(
                        f"Experiment: {reader.params['prefix']} already evaluated. Skipping.",
                        "yellow")
                    continue
                utils.cprint(
                    f"Evaluating experiment: {reader.params['prefix']}.", "green")

                date = reader.params['suffix']
                max_test_acc = max(train_acc[i])
                max_val_acc = max(val_acc[i])
                max_epoch = train_epochs[i][-1]
                utils.set_seed(reader.params['seed'])
                orig_class_labels = reader.params['class_labels']
                expected_labels = reader.params['class_labels'] if isinstance(
                    reader.params['class_labels'][0], int) else [i for i in range(len(reader.params['class_labels']))]
                while len(reader.params['class_labels']) < 3:
                    reader.params['class_labels'].append("-")
                if ib_dataset_param is None:
                    _, _, test_dataloader, ib_dataset = init_dataset(reader.params)
                else:
                    ib_dataset = ib_dataset_param
                reader.params['class_labels'] = [
                    label for label in reader.params['class_labels'] if label != '-']

                model = load_model(reader)
                test_patient_loops, test_patient_labels = ib_dataset.get_test_patient_loops()
                class_names = reader.params['class_labels'] if reader.params['class_labels'] is not None else [
                    "healthy", "COPD", "ILD"]
                label_to_idx = {name: idx for idx,
                                name in enumerate(class_names)}
                test_patient_labels = [label_to_idx[label]
                                       for label in test_patient_labels]
                if len(test_patient_loops) == 0:
                    utils.cprint(
                        f"No test data found for experiment: {reader.params['prefix']}. Skipping evaluation.",
                        "yellow")
                    continue

                patient_predictions_soft_vote = []
                patient_predictions_median = []
                all_loop_predictions = []
                all_loop_labels = []

                device = next(model.parameters()).device

                with torch.no_grad():
                    if reader.params['test_aggregation'] in ["majority_vote", "none", "all"]:
                        for patient_id, loops in enumerate(test_patient_loops):
                            patient_loop_probs = []

                            for loop in loops:
                                loop = ib_dataset.transform(loop, {"augment": False})[
                                    0].unsqueeze(0).to(device)
                                raw_output = model(loop)
                                probs = get_probabilities(raw_output)
                                patient_loop_probs.append(probs)

                                if reader.params['test_aggregation'] in ["none", "all"]:
                                    pred_label = torch.argmax(
                                        probs, dim=1).item()
                                    all_loop_predictions.append(pred_label)
                                    all_loop_labels.append(
                                        test_patient_labels[patient_id])

                            if reader.params['test_aggregation'] in ["majority_vote", "all"]:
                                if len(patient_loop_probs) > 0:
                                    all_probs_tensor = torch.cat(
                                        patient_loop_probs, dim=0)
                                    avg_probs = torch.mean(
                                        all_probs_tensor, dim=0)
                                    final_patient_pred = torch.argmax(
                                        avg_probs).item()
                                    patient_predictions_soft_vote.append(
                                        final_patient_pred)
                                else:
                                    patient_predictions_soft_vote.append(-1)

                    if reader.params['test_aggregation'] in ["median_loop", "all"]:
                        for patient_id, loops in enumerate(test_patient_loops):
                            median_transform = MedianLoops()
                            loop_group = median_transform(loops)
                            loop = ib_dataset.transform(loop_group[0], {"augment": False})[
                                0].unsqueeze(0).to(device)
                            raw_output = model(loop)
                            _, predicted_label = raw_output.max(1)
                            patient_predictions_median.append(
                                predicted_label.item())

                if reader.params['test_aggregation'] in ["majority_vote", "all"]:
                    correct = sum(1 for i in range(len(test_patient_labels))
                                  if test_patient_labels[i] ==
                                  patient_predictions_soft_vote[i])
                    accuracy = (correct / len(test_patient_labels)) * 100
                    cm = confusion_matrix(test_patient_labels,
                                          patient_predictions_soft_vote,
                                          labels=expected_labels)
                    results_data.append({
                        "accuracy": accuracy,
                        "confusion_matrix": cm,
                        "aggregation_method": "majority_vote",
                        "model_label": reader.params['prefix'],
                        "max_test_acc": max_test_acc,
                        "max_val_acc": max_val_acc,
                        "epoch": max_epoch,
                        "date": date
                    })
                    cprint(
                        f"The MAJORITY VOTE (Soft Vote) accuracy of experiment: {reader.params['prefix']} is: {accuracy:.2f}",
                        "magenta")

                if reader.params['test_aggregation'] in ["median_loop", "all"]:
                    correct = sum(
                        1 for i in range(len(test_patient_labels))
                        if test_patient_labels[i] == patient_predictions_median[i])
                    accuracy = (correct / len(test_patient_labels)) * 100
                    cm = confusion_matrix(test_patient_labels,
                                          patient_predictions_median,
                                          labels=expected_labels)
                    results_data.append({
                        "accuracy": accuracy,
                        "confusion_matrix": cm,
                        "aggregation_method": "median_loop",
                        "model_label": reader.params['prefix'],
                        "max_test_acc": max_test_acc,
                        "max_val_acc": max_val_acc,
                        "epoch": max_epoch,
                        "date": date
                    })
                    cprint(
                        f"The MEDIAN LOOP accuracy of experiment: {reader.params['prefix']} is: {accuracy:.2f}",
                        "magenta")

                if reader.params['test_aggregation'] in ["none", "all"]:
                    correct = sum(1 for i in range(len(all_loop_predictions))
                                  if all_loop_predictions[i] == all_loop_labels[i])
                    accuracy = (correct / len(all_loop_predictions)) * 100
                    cm = confusion_matrix(
                        all_loop_labels, all_loop_predictions, labels=expected_labels)
                    results_data.append({
                        "accuracy": accuracy,
                        "confusion_matrix": cm,
                        "aggregation_method": "loop_wise",
                        "model_label": reader.params['prefix'],
                        "max_test_acc": max_test_acc,
                        "max_val_acc": max_val_acc,
                        "epoch": max_epoch,
                        "date": date
                    })
                    cprint(
                        f"The LOOP-WISE accuracy of experiment: {reader.params['prefix']} is: {accuracy:.2f}",
                        "magenta")

            except Exception as e:
                utils.cprint(
                    f"Error evaluating {reader.params['prefix']}: {str(e)}", "red")
                utils.cprint(f"Traceback:\n{traceback.format_exc()}", "red")
                continue

    except Exception as e:
        utils.cprint(
            f"Error evaluating {reader.params['prefix']}: {str(e)}", "red")
        utils.cprint(f"Traceback:\n{traceback.format_exc()}", "red")

    if results_data:
        print(
            f"Saving test results to {os.path.join(output_dir, 'test_results.csv')}")
        results_df = pd.DataFrame(results_data)
        results_df['confusion_matrix_str'] = results_df['confusion_matrix'].apply(
            lambda cm: str(cm.tolist())[1:-1])
        results_df = results_df.drop(columns=['confusion_matrix'])
        results_df.to_csv(os.path.join(
            output_dir, "test_results.csv"), index=False)
        return results_df
    else:
        return pd.DataFrame()


def load_model(reader: LogReader) -> torch.nn.Module:
    model_name = reader.params['best_model_filename']
    if not os.path.exists(model_name):
        model_name = os.path.join(
            reader.params['output_dir'],
            f"{reader.params['prefix']}_{reader.params['model']}_model_backup_{reader.params['suffix']}.pt")
        if not os.path.exists(model_name):
            raise FileNotFoundError(
                f"Model file '{model_name}' does not exist.")
    device = torch.device(
        "cuda" if reader.params['gpu'] else "cpu")
    model = get_model(reader.params)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_dataset(reader: LogReader, type: str = 'test') -> DataLoader[IBDataset]:
    utils.set_seed(reader.params['seed'])
    train_dataloader, validation_dataloader, test_dataloader, ib_dataset = init_dataset(
        reader.params)
    train_features, train_labels = next(iter(train_dataloader))
    reader.params['class_labels'] = ib_dataset.class_labels
    reader.params['num_classes'] = len(reader.params['class_labels'])
    if type == 'train':
        return train_dataloader
    elif type == "val":
        return validation_dataloader
    elif type == "test":
        return test_dataloader
    else:
        raise ValueError(
            f"Unknown dataset type: {type}. Use 'train', 'val', or 'test'.")


def get_train_val_values(logreaders: list[LogReader],
                         value_str: str = "loss") -> tuple[list[list[int]],
                                                           list[list[float]],
                                                           list[list[int]],
                                                           list[list[float]]]:
    """
    Extracts the training and validation loss values from the log readers.
    Returns the epochs and corresponding loss values for both training and validation.
    Args:
    logreaders: list[LogReader]
    value_str: the str for metrics that appear in both train and validation
    """
    assert len(logreaders) > 0, "No log readers provided"
    train_acc = []
    val_acc = []
    train_epochs = []
    val_epochs = []
    for i, reader in enumerate(logreaders):
        try:
            train_loss_values: list = reader.event_to_list(
                'train_' + value_str, [str, int, str, float])
            train_acc.append(train_loss_values[3])
            train_epochs.append(train_loss_values[1])
            acc_loss_values: list = reader.event_to_list(
                'val_' + value_str, [str, int, str, float])
            val_acc.append(acc_loss_values[3])
            val_epochs.append(acc_loss_values[1])
        except Exception as e:
            print(f"Error processing log reader {reader.path}: {e}")
            train_acc.append([])
            val_acc.append([])
            train_epochs.append([])
            val_epochs.append([])

    return train_epochs, train_acc, val_epochs, val_acc


def plot_train_val_values(
        logreaders: list[LogReader],
        value_str: str = 'loss', labels: Optional[list[str]] = None,
        save_path: Optional[str] = None):
    train_epochs, train_acc, val_epochs, val_acc = get_train_val_values(
        logreaders, value_str=value_str)
    visualize.init_plt()
    visualize.compare_train_val(train_epochs,
                                train_acc,
                                val_epochs,
                                val_acc,
                                labels=labels,
                                save_path=save_path,
                                ylabel=value_str)


def check_accuracy_no_eval(
        scores, y, class_labels: Optional[list[str]] = None) -> float:
    num_correct = 0
    num_samples = 0
    if class_labels is None:
        class_names = ["healthy", "COPD", "ILD", "asthma'"]
    else:
        class_names = class_labels
    # Create a mapping from class names to indices
    # This mapping is used to convert string labels to integer indices
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    _, predictions = scores.max(1)
    target = torch.tensor([label_to_idx[y_]
                           for y_ in y], device=scores.device)
    # Count correct predictions
    num_correct += (predictions == target).sum().item()
    num_samples += predictions.size(0)  # Count total samples

    # Calculate accuracy
    accuracy = float(num_correct) / float(num_samples) * 100
    # print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    return accuracy


def calculate_detailed_csv(input_path, output_path, aggregate_by_seed=False):
    # 1. Load the dataset
    df = pd.read_csv(input_path)

    def parse_cm(cm_str):
        """Converts string matrix representation to numpy array."""
        try:
            # Matches your format: "[42, 2, 17], [0, 35, 6]..."
            formatted_str = "[" + str(cm_str) + "]"
            return np.array(ast.literal_eval(formatted_str))
        except:
            return None

    def get_row_metrics(cm):
        """Calculates class-wise and aggregate metrics from a confusion matrix."""
        if cm is None or not isinstance(cm, np.ndarray):
            return pd.Series(dtype=float)

        num_classes = cm.shape[0]
        total = np.sum(cm)
        metrics = {}
        precisions, recalls, f1s, supports = [], [], [], []

        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = total - (tp + fp + fn)

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            a = (tp + tn) / total if total > 0 else 0
            metrics[f'class_{i}_acc'] = a
            metrics[f'class_{i}_f1'] = f
            metrics[f'class_{i}_recall'] = r
            metrics[f'class_{i}_precision'] = p

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            supports.append(np.sum(cm[i, :]))

        metrics['macro_f1'] = np.mean(f1s)
        metrics['macro_recall'] = np.mean(recalls)
        metrics['weighted_f1'] = np.average(f1s, weights=supports)

        return pd.Series(metrics)

    # 2. Generate per-row metrics
    cms = df['confusion_matrix_str'].apply(parse_cm)
    metrics_df = cms.apply(get_row_metrics)
    full_df = pd.concat([df, metrics_df], axis=1)

    # 3. Handle Seed Aggregation
    if aggregate_by_seed:
        # Regex to find '_seed_1_' or '_seed_123_' and remove it
        # This creates a 'config_group' column to group identical models
        full_df['config_group'] = full_df['model_label'].apply(
            lambda x: re.sub(r'_seed_\d+_', '_seed_aggregated_', str(x))
        )

        # Define which columns to average (all numeric ones)
        # We exclude epoch if you don't want to average that
        numeric_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()

        # Group by the new config name and aggregation method
        final_df = full_df.groupby(['config_group', 'aggregation_method'])[
            numeric_cols].mean().reset_index()

        # Rename config_group back to model_label for consistency
        final_df = final_df.rename(columns={'config_group': 'model_label'})
    else:
        final_df = full_df

    final_df.to_csv(output_path, index=False)
    print(
        f"Success! File saved to {output_path} (Seed Aggregation: {aggregate_by_seed})")
