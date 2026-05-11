from pathlib import Path
import torch
import numpy as np
from typing import Optional, Callable
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping on validation loss (min),
    checkpointing on validation accuracy (max).
    """

    def __init__(
        self,
        patience: int = 7,
        delta_loss: float = 0.0,
        delta_acc: float = 0.0,
        path: str | Path = "checkpoint.pt",
        verbose: bool = False,
        trace_func=print,
        min_epochs: int = 15,
    ):
        self.patience = patience
        self.delta_loss = delta_loss
        self.delta_acc = delta_acc
        self.path = path
        self.verbose = verbose
        self.trace_func = trace_func
        self.min_epochs = min_epochs

        self.counter = 0
        self.early_stop = False

        self.best_val_loss = np.inf
        self.best_val_acc = -np.inf
        self.saved = False

    def __call__(self, val_loss: float, val_acc: float, model: nn.Module, epoch: int = 0):
        """
        Call method to check for early stopping and save checkpoints. Handles both validation loss and accuracy.
        Validation loss is used for early stopping (minimization), while validation accuracy is used for checkpointing (maximization).

        :param val_loss: Validation loss
        :param val_acc: validation accuracy
        :param model: Model to save
        :param epoch: Epoch number
        """
        self.saved = False

        # early stopping on loss
        if val_loss < self.best_val_loss - self.delta_loss:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs:
                self.early_stop = True

        # checkpoint on accuracy if improved beyond delta_acc threshold
        if val_acc > self.best_val_acc + self.delta_acc:
            self.best_val_acc = val_acc
            self.save_checkpoint(val_acc, model)

    def save_checkpoint(self, val_acc: float, model: nn.Module):
        """
        Saves model when validation accuracy improves. 

        Args:
            val_acc (float): Validation accuracy that triggered the checkpoint
            model (nn.Module): Model to save
        """
        if self.verbose:
            self.trace_func(
                f"Validation accuracy improved to {val_acc:.4f}. Saving model."
            )
        torch.save(model.state_dict(), self.path)
        self.saved = True

    def load_best_model(self, model: nn.Module):
        """
        Loads the best model from the checkpoint path.

        Args:
            model (nn.Module): Model architecture to load the weights into
        """
        model.load_state_dict(torch.load(self.path))
        self.trace_func(f"Loaded best model from {self.path}")

    def set_state(self, counter_state: int = 0, best_loss: float = np.inf, best_acc: float = -np.inf):
        """
        Sets the internal state of the early stopping mechanism, allowing for resuming from a specific point.

        Args:
            counter_state (int, optional): Counter state to resume from. Defaults to 0.
            best_loss (float, optional): Best validation loss to resume from. Defaults to np.inf.
            best_acc (float, optional): Best validation accuracy to resume from. Defaults to -np.inf.
        """
        self.counter = counter_state
        self.best_val_loss = best_loss
        self.best_val_acc = best_acc
