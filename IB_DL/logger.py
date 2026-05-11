from pathlib import Path
from typing import Any, List
import ast
import json


class Logger:
    def __init__(self, path: Path | str) -> None:
        self.path = path

    def open(self):
        self.file = open(self.path, "a")
        assert self.file, "Could not open file"
        return self.file

    def close(self):
        self.file.flush()
        self.file.close()

    def write(self, str):
        self.file.write(str)

    def writeline(self, str):
        self.file.write(str + "\n")

    def write_params(self, params: dict):
        for k, v in params.items():
            temp = f"param, {k}, {v}"
            self.writeline(temp)

    def write_event(self, key: str, params: dict):
        temp = key
        for k, v in params.items():
            temp += f", {k}, {v}"
        self.writeline(temp)


class LogReader:
    def __init__(self, path: Path | str) -> None:
        self.path = path
        self.params: dict[str, Any] = {}

    def read(self):
        with open(self.path, "r") as file:
            self.lines = file.readlines()

    def read_params(self):
        params = {}
        for line in self.lines:
            if line.startswith("param"):
                parts = line.replace(" ", "").replace("\n", "").split(",")
                if parts[1] == "log_filename" or parts[1] == "best_model_filename" or parts[1] == "prefix" and len(parts) > 3:
                    params[parts[1]] = ", ".join(parts[2:])
                    continue
                if '[' in parts[2]:
                    trans_list_str = ",".join(parts[2:])
                    # Replace numpy types with Python equivalents
                    trans_list_str = trans_list_str.replace(
                        "np.float64(", "(").replace("np.int64(", "(")
                    try:
                        trans_list = ast.literal_eval(trans_list_str)
                    except Exception:
                        trans_list = []
                    params[parts[1]] = trans_list
                else:
                    params[parts[1]] = self.read_value(parts[2])
        self.params: dict[str, Any] = params

        return params

    def read_resume_params(self):
        """
        Get parameters for resuming training from log file.
         :return    
         new_lr: float, learning rate to resume from
         last_epoch: int, epoch to resume from
         early_cnt: int, early stopping counter state
         best_acc: float, best validation accuracy so far
         best_loss: float, best validation loss so far
         lr_counter: int, learning rate scheduler counter state
        """
        best_loss, lr_counter, new_lr, early_cnt, last_epoch = self.get_scheduler_state()

        val_acc_events = self.read_event("val_acc")

        val_acc_list = [float(event.split(", ")[3])
                        for event in val_acc_events]
        best_acc = max(val_acc_list)
        print(
            f"Resuming from epoch {last_epoch} with learning rate {new_lr}",
            "highlighted_yellow")
        return new_lr, last_epoch, early_cnt, best_acc, best_loss, lr_counter

    def read_value(self, input: str):
        try:
            param_values = ast.literal_eval(input)
        except Exception:
            param_values = input
        return param_values

    def read_event(self, key: str):
        events = []
        for line in self.lines:
            if line.startswith(key):
                events.append(line.strip().split(", ", maxsplit=1)[-1])
        return events

    def get_train_loss(self):
        events: List[str] = self.read_event("train_loss")
        assert len(events) > 0
        assert events[0].count(", ") > 0
        assert events[0].count("epoch") > 0
        assert events[0].count("value") > 0
        epochs = []
        values = []
        for event in events:
            elements = event.split(", ")
            epochs.append(int(elements[1]))
            values.append(float(elements[3]))
        return epochs, values

    def event_to_list(self, key: str, dtypes: List[type] = [str, int, str, float]) -> List[List[Any]]:
        events = self.read_event(key)
        result = []
        for event in events:
            elements = event.split(", ")
            assert len(elements) == len(
                dtypes), f"Expected {len(dtypes)} elements, got {len(elements)}"
            converted = [dtype(e) for e, dtype in zip(elements, dtypes)]
            result.append(converted)
            # Transpose the result to get a list of elements for every event
        transposed = list(map(list, zip(*result)))
        return transposed

    def early_stop(self):
        return True

    def get_scheduler_state(self):
        """
        Parses the log to find:
        1. best_val_loss (for scheduler.best)
        2. num_bad_epochs (for scheduler.num_bad_epochs)
        """
        best_loss = float('inf')
        last_reset_epoch = 0  # Epoch where counter was last set to 0
        current_epoch = 0
        new_lr = self.params["learning_rate"]
        best_loss_epoch = 0

        for line in self.lines:
            if not line:
                continue
            row = line.strip().split(", ")
            event_type = row[0]

            if event_type == 'val_loss':
                current_epoch = int(row[2])
                val_loss = float(row[4])

                if val_loss < best_loss:
                    best_loss = val_loss
                    last_reset_epoch = current_epoch
                    best_loss_epoch = current_epoch

            elif event_type == 'lr_change':
                last_reset_epoch = current_epoch
                new_lr = float(row[4])

        num_bad_epochs = current_epoch - last_reset_epoch
        early_cnt = current_epoch - best_loss_epoch

        return best_loss, num_bad_epochs, new_lr, early_cnt, current_epoch

    def modify_param(self, param_name: str, new_value: Any) -> bool:
        """
        Modifies a parameter in the log file, handling open file states automatically.
        """
        file_path = Path(self.path)

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return False

        modified = False

        for i, line in enumerate(self.lines):
            if line.startswith("param,"):
                parts = line.split(", ", maxsplit=2)
                if len(parts) >= 2 and parts[1].strip() == param_name:
                    self.lines[i] = f"param, {param_name}, {new_value}\n"
                    modified = True
                    break

        if modified:
            with open(file_path, "w") as f:
                f.writelines(self.lines)

        return modified
