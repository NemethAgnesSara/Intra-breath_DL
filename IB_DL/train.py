import os
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader
from IB_DL.datasets import init_dataset
from IB_DL.eval import check_accuracy_no_eval
from IB_DL.models import get_model
from IB_DL.early_stop import EarlyStopping
from IB_DL.logger import Logger
from IB_DL.utils import cprint, print_epoch, set_seed, check_param_compatibility, get_optimizer
from IB_DL.config import DefaultLabels
from IB_DL import utils


class LabelCrossEntropyLoss(nn.Module):
    """
    Custom loss function that compares model outputs (logits) to string labels.
    Assumes the model outputs a tensor of shape (batch_size, num_classes)
    and the labels are a tuple/list of strings.

    """

    def __init__(
            self, class_labels: Optional[list[str]] = None,
            class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_names = class_labels if class_labels is not None else DefaultLabels().class_labels

        # Create a mapping from class names to integer indices
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, outputs: torch.Tensor, labels: tuple[str]) -> torch.Tensor:
        """
        Calculates Cross-Entropy Loss for string-based target labels.

        Args:
            outputs (torch.Tensor): Model outputs (logits) of shape (batch_size, num_classes).
            labels (tuple[str]): Ground truth labels as a tuple/list of strings.

        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
        # Convert string labels to integer indices and move to the device of the outputs
        target = torch.tensor([self.label_to_idx[label]
                              for label in labels], device=outputs.device)
        return self.ce(outputs, target)


def train_loop(model: nn.Module,
               dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               params: dict,
               lr_schedulers: Optional[Tuple[torch.optim.lr_scheduler.LinearLR,
                                             torch.optim.lr_scheduler.ReduceLROnPlateau]] = None,
               val_loader: Optional[DataLoader] = None,
               log_file: Optional[Logger] = None,
               resume: bool = False) -> None:
    """
    Executes the main training and validation epochs for the model.

    Args:
        model (nn.Module): The neural network model to be trained.
        dataloader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        params (dict): Dictionary containing training parameters (epochs, device, early_counter, etc.).
        lr_schedulers (tuple, optional): Tuple of (LinearLR, ReduceLROnPlateau) for warmup and decay.
        val_loader (DataLoader, optional): DataLoader for validation. If provided, validation is run.
        log_file (Logger, optional): Logger instance for recording metrics.
        resume (bool): If True, restores early stopping states to resume training seamlessly.
    """
    overfitting_counter = 0
    size = len(dataloader.dataset)  # type: ignore

    # Log initial dataset sizes
    if log_file and not resume:
        log_file.write_params({"train_size": size})
        if val_loader is not None:
            val_size = len(val_loader.dataset)  # type: ignore
            log_file.write_params({"val_size": val_size})

    # Prepare Labels and Criterion
    class_labels = params.get("class_labels") or DefaultLabels().class_labels
    criterion = LabelCrossEntropyLoss(class_labels=class_labels, class_weights=None)

    # Initialize Early Stopping
    early_stopping = EarlyStopping(
        patience=params["early_counter"],
        verbose=False,
        path=params["best_model_filename"],
        min_epochs=params.get("min_epochs", 0)
    )

    if resume:
        early_stopping.set_state(
            counter_state=params.get('early_last_state', 0),
            best_loss=params.get('best_val_loss', np.inf),
            best_acc=params.get('best_val_accuracy', -np.inf)
        )

    for epoch in range(params.get("start_epoch", 0), params["epochs"]):
        losses = {'Train': -1.0, 'Val': -1.0}
        accuracies = {'Train': -1.0, 'Val': -1.0}
        lr = optimizer.param_groups[0]['lr']
        early_counter = early_stopping.counter
        saved = False

        # Training Phase
        model.train()
        epoch_loss = 0.0
        train_all_preds = []
        train_all_labels = []

        start_time = time.time()

        for batch, (X, y) in enumerate(dataloader):
            print(
                f"Epoch {epoch+1}/{params['epochs']}, Batch {batch+1}/{len(dataloader)}",
                end='\r')
            X = X.to(params["device"])

            # Forward pass
            pred_train = model(X)
            loss = criterion(pred_train, y)

            # Optional L2 Regularization
            if hasattr(model, "l2_regularization"):
                bs = X.size(0)
                reg_fn_or_value = model.l2_regularization
                if callable(reg_fn_or_value):
                    reg = reg_fn_or_value(bs)
                else:
                    reg = reg_fn_or_value
                    if not isinstance(reg, torch.Tensor):
                        reg = torch.tensor(reg, device=loss.device, dtype=loss.dtype)
                loss = loss + reg

            if not torch.isfinite(loss):
                cprint(f"Loss is NaN at epoch {epoch}, batch {batch}", "error")
                return

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)

            # Detach to prevent future gradient calculations accumulating in memory
            train_all_preds.append(pred_train.detach())
            train_all_labels.extend(y)

        timing = time.time() - start_time
        avg_loss = epoch_loss / size
        losses['Train'] = avg_loss

        if log_file:
            log_file.write_event("train_loss", {"epoch": epoch, "value": avg_loss})

        # Calculate Training Accuracy
        train_all_preds = torch.cat(train_all_preds)
        train_acc = check_accuracy_no_eval(
            scores=train_all_preds, y=train_all_labels, class_labels=class_labels)
        accuracies['Train'] = train_acc

        if log_file:
            log_file.write_event("train_acc", {"epoch": epoch, "value": train_acc})

        # Validation Phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_all_preds = []
            val_all_labels = []

            with torch.no_grad():
                for batch, (X, y) in enumerate(val_loader):
                    print(
                        f"Epoch {epoch+1}/{params['epochs']}, Val Batch {batch+1}/{len(val_loader)}",
                        end='\r')
                    X = X.to(params["device"])

                    pred_val = model(X)
                    loss = criterion(pred_val, y)

                    val_loss += loss.item()
                    val_all_preds.append(pred_val.detach())
                    val_all_labels.extend(y)

            val_loss /= len(val_loader)
            losses['Val'] = val_loss

            if log_file:
                log_file.write_event("val_loss", {"epoch": epoch, "value": val_loss})

            # Calculate Validation Accuracy
            val_all_preds = torch.cat(val_all_preds)
            val_acc = check_accuracy_no_eval(
                scores=val_all_preds, y=val_all_labels, class_labels=class_labels)
            accuracies['Val'] = val_acc

            if log_file:
                log_file.write_event("val_acc", {"epoch": epoch, "value": val_acc})

            # Schedulers step based on epoch/validation
            if lr_schedulers is not None:
                current_lr = optimizer.param_groups[0]['lr']
                if epoch <= params.get("warmup_epochs", 0):
                    lr_schedulers[0].step()
                else:
                    lr_schedulers[1].step(val_loss)

                lr = optimizer.param_groups[0]['lr']
                if current_lr != lr and log_file:
                    log_file.write_event("lr_change", {"old": current_lr, "new": lr})

            # Early Stopping Check
            early_stopping(val_loss=val_loss, val_acc=val_acc, model=model, epoch=epoch)
            early_counter = early_stopping.counter

            if early_stopping.saved:
                saved = True
                if log_file:
                    log_file.write_event("saving", {"epoch": epoch})
                early_stopping.saved = False

            if early_stopping.early_stop:
                if log_file:
                    log_file.write_event("early_stop", {"epoch": epoch})
                break

        # End of epoch logging and checks
        if train_acc > 98:
            cprint(
                f"Training succeeded at epoch {epoch}. Train Acc: {train_acc:.2f}%, Val Acc: {accuracies['Val']:.2f}%",
                "yellow")
            break

        if val_loader is not None and train_acc > accuracies['Val'] + 15.0 and epoch > 15:
            overfitting_counter += 1
            if overfitting_counter >= 3:
                if log_file:
                    log_file.write_event("overfitting_stop", {"epoch": epoch})
                cprint(
                    f"Warning: Possible overfitting detected at epoch {epoch}. Train Acc: {train_acc:.2f}%, Val Acc: {accuracies['Val']:.2f}%",
                    "yellow")
                break
        else:
            overfitting_counter = 0

        if log_file:
            log_file.write_event("time", {"epoch": epoch, "value": timing})

        print_epoch(losses, accuracies, timing, epoch, lr, early_counter,
                    saved, print_level=params.get("log_level", "INFO"))


def train(params: Dict,
          resume: bool = False,
          train_dataloader: Optional[DataLoader] = None,
          validation_dataloader: Optional[DataLoader] = None,
          class_labels: Optional[list[str]] = None) -> None:
    """
    Setup and execution environment for the training process. Initializes data, 
    model, optimizer, and handles global exception safety and model rescuing.

    Args:
        params (Dict): Dictionary containing configurations (architecture, lr, epochs, etc.).
        resume (bool): If True, restores weights and states to resume from the last checkpoint.
        train_dataloader (DataLoader, optional): Pre-initialized train DataLoader.
        validation_dataloader (DataLoader, optional): Pre-initialized validation DataLoader.
        class_labels (list[str], optional): Explicit list of class labels to track.
    """
    set_seed(params["seed"])
    check_param_compatibility(params)

    print("Training model with the following parameters:")
    for k, v in params.items():
        cprint(f"  {k}: {v}", "blue")

    # Dataset Initialization
    if train_dataloader is None or validation_dataloader is None:
        train_dataloader, validation_dataloader, test_dataloader, ib_dataset = init_dataset(
            params)
        params["class_labels"] = ib_dataset.class_labels
    else:
        params["class_labels"] = class_labels

    params["num_classes"] = len(params["class_labels"])  # type: ignore

    print(
        f"Train dataloader size: {len(train_dataloader)}, Validation dataloader size: {len(validation_dataloader)}")

    params["device"] = "cuda" if params.get(
        "gpu", False) and torch.cuda.is_available() else "cpu"

    # Model Initialization
    nn_model = get_model(params)
    if resume:
        cprint("Resuming training from checkpoint.", "red")
        nn_model.load_state_dict(
            torch.load(
                params["best_model_filename"],
                map_location=params["device"]))

    nn_model = nn_model.to(params["device"])

    # Optimizer & Schedulers
    optimizer = get_optimizer(params["optimizer"], nn_model, params["learning_rate"])

    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, cooldown=1, threshold=0.01
    )
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=params["learning_rate"],
        total_iters=params.get("warmup_epochs", 0))

    if resume:
        scheduler2.best = params.get("best_val_loss", np.inf)
        scheduler2.num_bad_epochs = params.get("lr_scheduler_last_state", 0)
        scheduler2.cooldown_counter = 0
        if params.get("log_level") == "INFO":
            cprint(
                f"  [Resume] Scheduler Restored: Best: {scheduler2.best}, Bad Epochs: {scheduler2.num_bad_epochs}",
                "blue")

    utils.header = True
    log_file = None

    try:
        set_seed(params["seed"])
        log_file = Logger(params["log_filename"])
        log_file.open()

        if not resume:
            log_file.write_params(params=params)

        train_loop(
            model=nn_model,
            dataloader=train_dataloader,
            params=params,
            optimizer=optimizer,
            lr_schedulers=(scheduler1, scheduler2),
            val_loader=validation_dataloader,
            log_file=log_file,
            resume=resume
        )

    except (Exception, KeyboardInterrupt) as e:
        cprint(f"Training stopped: {e}", "error")
        cprint("Stack Traceback:", "yellow")
        traceback.print_exc()

        cprint("Saving model state before exiting...", "error")
        model_save_path = os.path.join(
            params["output_dir"],
            f"{params.get('prefix', 'chk')}_{params['model']}_model_backup_{params.get('suffix', 'err')}.pt")
        torch.save(nn_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        if log_file is not None:
            log_file.write_event("exception", {"type": str(type(e)), "message": str(e)})
            log_file.write_event("saving", {"epoch": -1})

    finally:
        if log_file is not None:
            log_file.close()

        if 'train_dataloader' in locals():
            del train_dataloader
        if 'validation_dataloader' in locals():
            del validation_dataloader
        if 'test_dataloader' in locals():
            del test_dataloader
        torch.cuda.empty_cache()
