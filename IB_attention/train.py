import json
import math
import os
import random
from typing import Any, Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader  # type: ignore
from IB_attention.datasets import Loop, init_dataset
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from IB_attention.eval import check_accuracy_no_eval, evaluate
from IB_attention.models import get_model
import torch.nn as nn
from torch.utils.data import Dataset
from IB_attention.early_stop import EarlyStopping
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from IB_attention.logger import Logger
import time
from IB_attention.utils import cprint, print_epoch, set_seed, check_param_compatibility
from IB_attention import utils
from IB_attention.utils import get_optimizer
from IB_attention.config import DefaultLabels
from torchinfo import summary
import traceback


class LabelCrossEntropyLoss(nn.Module):
    """
    Custom loss function that compares model outputs (logits) to string labels.
    Assumes the model outputs a tensor of shape (batch_size, num_classes)
    and the labels are a tuple/list of strings.
    """

    def __init__(self, class_labels: Optional[list[str]] = None, class_weigths: Optional[torch.Tensor] = None):
        super().__init__()
        if class_labels is None:
            self.class_names = DefaultLabels().class_labels
        else:
            self.class_names = class_labels
        # Create a mapping from class names to indices
        # This mapping is used to convert string labels to integer indices
        self.label_to_idx = {name: idx for idx,
                             name in enumerate(self.class_names)}
        self.ce = nn.CrossEntropyLoss(weight=class_weigths)

    def forward(self, outputs: torch.Tensor, labels: tuple[str]) -> torch.Tensor:
        """
        Args:
            outputs (torch.Tensor): Model outputs of shape (batch_size, num_classes).
            labels (tuple[str]): Labels as a tuple/list of strings.
        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
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
               resume: bool = False
               ) -> None:
    """
    Training loop for the model.
    This function trains the model for a specified number of epochs using the provided dataloader.
    It computes the loss using a custom cross-entropy loss function that handles string labels.
    It also checks the accuracy of the model on the training set and optionally on a test set after each epoch.

    Args:
        model (nn.Module): The neural network model to be trained.
        dataloader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        params (dict): Dictionary containing training parameters such as epochs, device, class_labels, etc.
        lr_scheduler (Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]): Learning rate scheduler to adjust the learning rate during training.
        val_loader (Optional[DataLoader]): DataLoader for the validation dataset. If provided, validation will be performed after each epoch.
        log_file (Optional[Logger]): Logger for recording training progress and metrics. 
        resume (bool): If True, training will resume from the last checkpoint. The checkpoint should contain the model state, optimizer state, and learning rate scheduler state.
    """
    overfitting_counter = 0
    size = len(dataloader.dataset)  # type: ignore
    if log_file and not resume:
        log_file.write_params({"train_size": size})
        if val_loader is not None:
            val_size = len(val_loader.dataset)  # type: ignore
            log_file.write_params({"val_size": val_size})
    model.train()
    train_labels = []
    for batch, (X, y) in enumerate(dataloader):
        train_labels.extend(y)  # type: ignore
    
    class_labels = params["class_labels"] if params["class_labels"] is not None else DefaultLabels(
    ).class_labels

    criterion = LabelCrossEntropyLoss(
        class_labels=params["class_labels"], class_weigths=None)

    early_stopping = EarlyStopping(patience=params["early_counter"], verbose=False, path=params["best_model_filename"],
                                   min_epochs=params["min_epochs"])
    if resume:
        early_stopping.set_state(
            counter_state=params.get('early_last_state', 0),
            best_loss=params.get('best_val_loss', np.inf),
            best_acc=params.get('best_val_accuracy', -np.inf))
        
    for epoch in range(params.get("start_epoch", 0), params["epochs"]):
        losses, accuracies, timing, epoch_id, lr, early_counter, saved = {
            'Train': -1.0, 'Val': -1.0}, {'Train': -1.0, 'Val': -1.0}, 0, epoch, optimizer.param_groups[0]['lr'], early_stopping.counter, False
        
        # training phase
        epoch_loss = 0.0
        train_all_preds = []
        train_all_labels = []
        start_time = time.time()

        dataloader_iter = iter(dataloader)
        for batch in range(len(dataloader)):
            
            try:
                X, y = next(dataloader_iter)
            except StopIteration:
                break
            
            print(
                f"Epoch {epoch+1}/{params['epochs']}, Batch {batch+1}/{len(dataloader)}", end='\r')
            X, y = X.to(params["device"]), y  
            
            # Forward pass
            pred_train = model(X)
            loss = criterion(pred_train, y)

            # Regularization
            if hasattr(model, "l2_regularization"):
                bs = X.size(0)
                reg_fn_or_value = model.l2_regularization
                if callable(reg_fn_or_value):
                    reg = reg_fn_or_value(bs)
                else:
                    reg = reg_fn_or_value
                    if not isinstance(reg, torch.Tensor):
                        reg = torch.tensor(
                            reg, device=loss.device, dtype=loss.dtype)
                loss = loss + reg

            if not torch.isfinite(loss):
                cprint(f"Loss is NaN at epoch {epoch}, batch {batch}", "error")
                print(f"epoch loss: {epoch_loss}, size: {size}")
                print("loss:", loss)
                print("pred_train:", pred_train)
                return

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)

            # Detach to prevent future gradient calculations
            train_all_preds.append(pred_train.detach())
            train_all_labels.append(y)

        end_time = time.time()
        del dataloader_iter

        if params["log_level"] == "INFO":
            timing = end_time - start_time

        avg_loss = epoch_loss / size
        losses['Train'] = avg_loss

        if log_file:
            log_file.write_event(
                "train_loss", {"epoch": epoch, "value": avg_loss})

        # validation phase
        if val_loader is not None:
            val_all_preds = []
            val_all_labels = []
            val_loss = 0.0
            model.eval()

            with torch.no_grad():
                dataloader_iter = iter(val_loader)
                for batch in range(len(val_loader)):
                    try:
                        X, y = next(dataloader_iter)
                    except StopIteration:
                        break

                    print(
                        f"Epoch {epoch+1}/{params['epochs']}, Val Batch {batch+1}/{len(val_loader)}", end='\r')

                    X, y = X.to(params["device"]), y
                    
                    # Forward pass
                    pred_val = model(X)

                    loss = criterion(pred_val, y)
                    val_loss += loss.item()
                    val_all_preds.append(pred_val.detach())
                    val_all_labels.append(y)

            del dataloader_iter
            
            val_loss /= len(val_loader)  # type: ignore
            losses['Val'] = val_loss

            if log_file:
                log_file.write_event(
                    "val_loss", {"epoch": epoch, "value": val_loss})

            # Update learning rate schedulers (for warmup and plateau)
            if lr_schedulers is not None:
                current_lr = optimizer.param_groups[0]['lr']
                if epoch <= params["warmup_epochs"]:
                    lr_schedulers[0].step()
                else:
                    lr_schedulers[1].step(val_loss)

                lr = optimizer.param_groups[0]['lr']
                
                if current_lr != optimizer.param_groups[0]['lr']:
                    if log_file:
                        log_file.write_event("lr_change", {"old": current_lr,
                                                           "new": optimizer.param_groups[0]['lr']})

        train_all_preds = torch.cat(train_all_preds)
        train_all_labels = [
            item for sublist in train_all_labels for item in sublist]
        train_acc = check_accuracy_no_eval(
            scores=train_all_preds, y=train_all_labels, class_labels=params["class_labels"])
        
        if log_file:
            log_file.write_event(
                "train_acc", {"epoch": epoch, "value": train_acc})
        
        accuracies['Train'] = train_acc
        
        if val_loader is not None:
            val_all_preds = torch.cat(val_all_preds)  # type: ignore
            val_all_labels = [
                item for sublist in val_all_labels for item in sublist]
            val_acc = check_accuracy_no_eval(
                scores=val_all_preds, y=val_all_labels, class_labels=params["class_labels"])
            accuracies['Val'] = val_acc
            
            if log_file:
                log_file.write_event(
                    "val_acc", {"epoch": epoch, "value": val_acc})
            
            early_stopping(val_loss=val_loss, val_acc=val_acc,
                           model=model, epoch=epoch)
            early_counter = early_stopping.counter
            
            if early_stopping.early_stop:
                if log_file:
                    log_file.write_event("early_stop", {"epoch": epoch})
                break

            model.train()  

            if early_stopping.saved:
                saved = True
                if log_file:
                    log_file.write_event("saving", {"epoch": epoch})
                early_stopping.saved = False  

        if train_acc > 98:
            cprint(
                f"Training succeeded at epoch {epoch}. Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%", "yellow")
            break

        if train_acc > val_acc + 15.0 and epoch > 15:
            overfitting_counter += 1
            if overfitting_counter >= 3:
                if log_file:
                    log_file.write_event("overfitting_stop", {"epoch": epoch})
                cprint(
                    f"Warning: Possible overfitting detected at epoch {epoch}. Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%", "yellow")
                break
        else:
            overfitting_counter = 0
        
        if log_file:
            log_file.write_event(
                "time", {"epoch": epoch, "value": timing})
        print_epoch(losses, accuracies, timing, epoch_id, lr,
                    early_counter, saved, print_level=params["log_level"])


def train(params: Dict, 
          resume: bool = False, 
          train_dataloader : Optional[DataLoader]=None, 
          validation_dataloader : Optional[DataLoader]=None, 
          class_labels : Optional[list[str] | str]=None) -> None:
    """
    Training function that sets up the training environment, initializes the model, optimizer, and learning rate schedulers, and calls the training loop.
    Args:
      params (Dict): A dictionary containing all the necessary parameters for training, such as model architecture, learning rate, number of epochs, batch size, etc.
      resume (bool): If True, the training will resume from the last checkpoint. The checkpoint should contain the model state, optimizer state, and learning rate scheduler state.
      train_dataloader (Optional[DataLoader]): An optional DataLoader for the training dataset. If not provided, it will be initialized using the parameters in the `params` dictionary.
      validation_dataloader (Optional[DataLoader]): An optional DataLoader for the validation dataset. If not provided, it will be initialized using the parameters in the `params` dictionary.
      class_labels (Optional[list[str] | str]): An optional list of class labels. If not provided, it will be initialized using the parameters in the `params` dictionary or default labels.
    """
    set_seed(params["seed"])

    check_param_compatibility(params)

    print("Training model with the following parameters:")
    for k, v in params.items():
        cprint(f"  {k}: {v}", "blue")

    if train_dataloader is None or validation_dataloader is None:
        train_dataloader, validation_dataloader, test_dataloader, ib_dataset = init_dataset(
            params)
        params["class_labels"] = ib_dataset.class_labels
    else:
        params["class_labels"] = class_labels
    params["num_classes"] = len(params["class_labels"]) # type: ignore

    print(
        f"Train dataloader size: {len(train_dataloader)}, Validation dataloader size: {len(validation_dataloader)}") # type: ignore

    params["device"] = "cuda" if params["gpu"] else "cpu"

    nn_model = get_model(params)
    if resume:
        cprint("Resuming training from checkpoint.", "red")
        nn_model.load_state_dict(torch.load(
            params["best_model_filename"], map_location=params["device"]))

    nn_model = nn_model.to(params["device"])

    optimizer = get_optimizer(
        params["optimizer"], nn_model, params["learning_rate"])

    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, cooldown=1, threshold=0.01
    )

    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=params["learning_rate"], total_iters=params["warmup_epochs"])
    
    if resume:
        scheduler2.best = params["best_val_loss"]
        scheduler2.num_bad_epochs = params["lr_scheduler_last_state"]
        scheduler2.cooldown_counter = 0
        if params["log_level"] == "INFO":
            cprint(
                f"  [Resume] Scheduler Restored: Best: {scheduler2.best}, Bad Epochs: {scheduler2.num_bad_epochs}", "blue")
   
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
            writer=None,
            resume=resume
        )

    except (Exception, KeyboardInterrupt) as e:
        error_appeared = True
        cprint(f"Training stopped: {e}", "error")
        cprint("Stack Traceback:", "yellow")
        traceback.print_exc()
        cprint("Saving model state before exiting...", "error")
        # Save the model state
        model_save_path = os.path.join(
            params["output_dir"], f"{params['prefix']}_{params['model']}_model_backup_{params['suffix']}.pt")
        torch.save(nn_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        if log_file is not None:
            log_file.write_event(
                "exception", {"type": str(type(e)), "message": str(e)})
            log_file.write_event("saving", {"epoch": -1})

    finally:
        if log_file is not None:
            log_file.close()
        del train_dataloader
        del validation_dataloader
        del test_dataloader
        torch.cuda.empty_cache()
