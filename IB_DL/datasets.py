from collections import defaultdict
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union  # type: ignore
import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torch.utils.data import WeightedRandomSampler

from IB_DL.utils import seed_worker, set_seed, MEASURENT_FILES
from torchvision.transforms import v2


class Loop:
    def __init__(self,
                 patient_id: str,
                 loop_id: str,
                 volume: List[float],
                 flow: List[float],
                 resistance: List[float],
                 reactance: List[float]) -> None:
        """ 
        Loop class representing a single loop of respiratory mechanics data for a patient. Each loop is defined by a unique patient_id and loop_id, and contains the volume, flow, resistance, and reactance data for that loop.

        Args:
            patient_id (str): patient ID to which the loop belongs.
            loop_id (str): unique identifier for the loop.
            volume (List[float]): list of volume values.
            flow (List[float]): list of flow values.
            resistance (List[float]): list of resistance values.
            reactance (List[float]): list of reactance values.
        """
        self.patient_id: str = patient_id
        self.loop_id: str = loop_id
        self.volume: NDArray[np.float32] = np.array(volume, dtype=np.float32)
        self.flow: NDArray[np.float32] = np.array(flow, dtype=np.float32)
        self.resistance: NDArray[np.float32] = np.array(resistance, dtype=np.float32)
        self.reactance: NDArray[np.float32] = np.array(reactance, dtype=np.float32)


class IBDataset(Dataset[tuple[Loop, str]]):
    """
    A custom PyTorch Dataset for handling respiratory loops.

    This dataset can operate in two modes based on `loop_set`:
    1. Patient-level (loop_set=False): Returns lists of all loops belonging to a single patient.
    2. Loop-level (loop_set=True): Returns individual loops as standalone samples.
    """

    def __init__(self,
                 csv_file: str | Path | list[str | Path],
                 transform: Optional[Callable[[Any, Optional[dict]], Any]] = None,
                 target_transform: Optional[Callable[[Any, Optional[dict]], Any]] = None,
                 num_of_rows: Optional[int] = None,
                 class_labels: Optional[str] = None,
                 loop_set: bool = True,
                 test_aggregation: str = "none") -> None:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callabla, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
            num_of_rows (int, optional): Number of rows to read from the csv file. If None, reads all rows.
            class_labels (list[str], optional): List of class labels to map the labels in the dataset.
            loop_set (bool): If True, the dataset will be treated as a set of loops, otherwise as a set of patients.
            test_aggregation (str): Defines how test set predictions are aggregated (e.g., "none", "median_loop").

        """
        if not isinstance(csv_file, list):
            csv_file = [csv_file]

        df_list = []
        for file in csv_file:
            df_list.append(pd.read_csv(str(file), nrows=num_of_rows))
        df = pd.concat(df_list, ignore_index=True)

        # Handle label mapping
        mapping_dict = {}

        if class_labels is not None:
            assert len(class_labels) >= len(df['label'].unique())
            mapping_dict.update({i: class_labels[i]
                                for i in range(0, len(df['label'].unique()))})
            df['label'] = df['label'].map(mapping_dict)
            df = df[df['label'] != '-']
            self.class_labels = [
                label for label in mapping_dict.values() if label != '-']
        else:
            # baseline class labels, can be overridden by providing class_labels argument
            self.class_labels = ["healty",  "COPD", "ILD"]
            mapping_dict = {
                0: "healty", 1: "COPD", 2: "ILD"}
            df['label'] = df['label'].map(mapping_dict)

        # Group raw dataframe rows into structured Loop objects organized by patient
        self.patient_loops, self.annotations = IBDataset.create_loops(
            df, df["label"])
        self.transform: Optional[Callable[[
            Any, Optional[dict]], Any]] = transform
        self.target_transform: Optional[Callable[[
            Any, Optional[dict]], Any]] = target_transform
        self.loop_set = loop_set
        self.test_aggregation = test_aggregation

        self.extra_params = {}
        self.train_indices: Optional[Sequence[int]] = None

        # Pre-calculate dataset statistics and flatten structure if needed
        self.get_info()
        self.create_loop_dataset()

    def __len__(self) -> int:
        """
        Returns: 
            Int: number of loops in the dataset.
        """
        if self.loop_set:
            return len(self.all_loops)
        return len(self.patient_loops)

    def __getitem__(self, idx) -> tuple[Union[Loop, np.ndarray, Any], str]:
        """
        Gets the item at the specified index. Performs transformations if specified.
        Args:
            idx (int): Index of the loop to retrieve.
        Returns:
            tuple: (loop, label) where loop is a Loop object and label is the corresponding annotation.
        """
        loops: Loop | List[Loop] = []
        if self.loop_set:
            loops = self.all_loops[idx]
            label: str = str(self.all_annotations[idx])
        else:
            loops = self.patient_loops[idx]
            label: str = str(self.annotations[idx])
        self.extra_params["augment"] = True if self.train_indices is not None and idx in self.train_indices else False
        if self.transform:
            loops = self.transform(loops, self.extra_params)
            loops = loops[0]  # type: ignore
        if self.target_transform:
            label = self.target_transform(label, self.extra_params)
        return loops, label

    def get_test_patient_loops(self) -> tuple[List[List[Loop]], List[str]]:
        """
        Returns:
            List[List[Loop]]: List of lists of loops for each patient.
        """
        return [self.patient_loops[i] for i in self.test_indices], [self.annotations[i] for i in self.test_indices]

    def get_class_labels(self) -> list[str]:
        """
        Returns:
            List[str]: Class labels in the dataset.
        """
        return self.class_labels

    def get_num_classes(self) -> int:
        """
        Returns:
            List[str]: Class labels in the dataset.
        """
        return len(self.class_labels)

    def get_info(self):
        self.num_of_patients = len(self.patient_loops)
        self.num_of_loops = sum(len(loops) for loops in self.patient_loops)
        self.num_of_loops_per_patient = [
            len(loops) for loops in self.patient_loops]
        self.idxs_per_classes = {}
        for label in self.class_labels:
            self.idxs_per_classes[label] = np.where(
                np.array(self.annotations) == label)[0]
        self.num_of_loops_per_class = {label: len(
            idxs) for label, idxs in self.idxs_per_classes.items()}

    def split_patient_dataset_indices(
            self, ttv_ratio: list[float] = [0.7, 0.2, 0.1]) -> tuple[
            Sequence[int],
            Sequence[int],
            Sequence[int]]:
        """
        Patient wise splits the dataset into train, validation, and test sets based on the specified ratio.
        Args:
            ttv_ratio (list[float]): List containing the ratios for train, validation, and test sets.
        Returns:
            tuple: Three lists of indices for train, validation, and test sets.
        """
        assert abs(sum(ttv_ratio) -
                   1.0) < 1e-6, "The sum of ttv_ratio must be 1."
        total_patients = self.num_of_patients

        permutation = np.random.permutation(total_patients)

        # 1. Group permuted patient indices by their class
        patients_by_class = defaultdict(list)
        for p_idx in permutation:
            class_label = self.annotations[p_idx]
            if class_label in self.class_labels:
                patients_by_class[class_label].append(p_idx)

        train_indices = []
        val_indices = []
        test_indices = []

        num_classes = self.get_num_classes()
        if num_classes == 0:
            raise ValueError("No class labels defined for stratification.")

        # 2. Distribute patients from each class into train, val, test
        for c in self.class_labels:
            class_patients = patients_by_class[c]
            num_class_patients = len(class_patients)

            test_count_for_class = round(num_class_patients * ttv_ratio[2])
            val_count_for_class = round(num_class_patients * ttv_ratio[1])

            test_count_for_class = min(
                test_count_for_class, num_class_patients)
            val_count_for_class = min(
                val_count_for_class, num_class_patients - test_count_for_class)
            train_count_for_class = num_class_patients - \
                test_count_for_class - val_count_for_class

            train_count_for_class = max(0, train_count_for_class)

            current_idx = 0
            test_indices.extend(
                class_patients[current_idx: current_idx + test_count_for_class])
            current_idx += test_count_for_class

            val_indices.extend(
                class_patients[current_idx: current_idx + val_count_for_class])
            current_idx += val_count_for_class

            # The rest go to train
            train_indices.extend(
                class_patients[current_idx: current_idx + train_count_for_class])

        self.train_indices = train_indices
        return train_indices, test_indices, val_indices

    def create_loop_dataset(self) -> None:
        all_loops = []
        all_annotations = []
        loop_to_patient = []
        for patient_idx, (patient_loops, ann) in enumerate(
                zip(self.patient_loops, self.annotations)):
            all_loops.extend(patient_loops)
            all_annotations.extend([ann] * len(patient_loops))
            loop_to_patient.extend([patient_idx] * len(patient_loops))
        self.all_loops = all_loops
        self.all_annotations = all_annotations
        self.loop_to_patient = loop_to_patient

    def creat_dataset_from_indices(self, indices: Sequence[int]) -> "IBDataset":
        from IB_DL.transformations import MedianLoops

        # Select the patient_loops and annotations for the given indices
        subset_patient_loops = [self.patient_loops[i] for i in indices]
        subset_annotations = [self.annotations[i] for i in indices]
        # Create a new IBDataset instance with the subset data
        dset = IBDataset.__new__(IBDataset)  # Bypass __init__
        dset.patient_loops = subset_patient_loops
        dset.annotations = subset_annotations
        dset.class_labels = self.class_labels
        dset.transform = v2.Compose(
            [MedianLoops(), self.transform])  # type: ignore
        # dset.transform.insert(0, {"name": "MedianLoops"})  # type: ignore
        dset.target_transform = self.target_transform
        dset.loop_set = False
        dset.extra_params = self.extra_params
        dset.train_indices = None
        dset.test_aggregation = self.test_aggregation
        dset.get_info()
        if dset.loop_set:
            dset.create_loop_dataset()
        return dset

    def split_data_subsets(self, ttv_ratio: list[float] = [0.7, 0.2, 0.1]) -> list[torch.utils.data.Subset]:
        train_indices, test_indices, val_indices = self.split_patient_dataset_indices(
            ttv_ratio)
        self.test_indices = test_indices
        # Keep the patient-level split (indices into ``patient_loops``) around so
        # the exact train/test/val partition can be exported, regardless of
        # loop_set mode.
        self.patient_split_indices: dict[str, list[int]] = {
            "train": list(train_indices),
            "test": list(test_indices),
            "val": list(val_indices),
        }

        if not self.loop_set:
            train_subset = torch.utils.data.Subset(self, train_indices)
            val_subset = torch.utils.data.Subset(self, val_indices)
            test_subset = torch.utils.data.Subset(self, test_indices)
        else:
            assert hasattr(
                self, 'all_loops'), "Call create_loop_dataset() before splitting data subsets."

            # Build a mapping from patient index to loop indices in all_loops
            patient_to_loop_indices = defaultdict(list)
            loop_idx = 0
            for patient_idx, patient_loops in enumerate(self.patient_loops):
                for _ in patient_loops:
                    patient_to_loop_indices[patient_idx].append(loop_idx)
                    loop_idx += 1

            # Get loop indices for each split
            train_loop_indices = [idx
                                  for p_idx in train_indices
                                  for idx in patient_to_loop_indices[p_idx]]
            train_subset = torch.utils.data.Subset(self, train_loop_indices)
            self.train_indices = train_loop_indices

            val_loop_indices = [
                idx for p_idx in val_indices for idx in patient_to_loop_indices[p_idx]]
            val_subset = torch.utils.data.Subset(self, val_loop_indices)

            if self.test_aggregation == "none" or "majority_vote" == self.test_aggregation or "all" == self.test_aggregation:
                test_loop_indices = [
                    idx for p_idx in test_indices
                    for idx in patient_to_loop_indices[p_idx]]
                test_subset = torch.utils.data.Subset(self, test_loop_indices)
            elif self.test_aggregation == "median_loop":
                test_set = self.creat_dataset_from_indices(test_indices)
                test_subset = torch.utils.data.Subset(
                    test_set, range(len(test_set)))

        return [train_subset, test_subset, val_subset]

    def patient_ids(self, indices: Sequence[int]) -> list:
        """Map patient indices (into ``patient_loops``) to their raw patient IDs.

        Values are coerced to native Python types so they serialise cleanly.
        """
        ids = []
        for i in indices:
            pid = self.patient_loops[i][0].patient_id
            ids.append(pid.item() if hasattr(pid, "item") else pid)
        return ids

    def save_patient_split(self, path: str | Path,
                           extra: Optional[dict] = None) -> dict:
        """Write the current train/test/val split as patient IDs to ``path`` (JSON).

        Requires :meth:`split_data_subsets` (or :meth:`split_patient_dataset_indices`
        via it) to have been called first so ``patient_split_indices`` exists.
        The saved lists preserve split membership; IDs within each split are
        sorted for stable diffing against another framework's split.
        """
        import json
        if not hasattr(self, "patient_split_indices"):
            raise RuntimeError(
                "No split available; call split_data_subsets() before save_patient_split().")
        split = {
            name: sorted(self.patient_ids(idxs))
            for name, idxs in self.patient_split_indices.items()
        }
        payload = {
            "class_labels": list(self.class_labels),
            "counts": {name: len(ids) for name, ids in split.items()},
            **split,
        }
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return payload

    def plot_histogram(self):
        """
        Plots a histogram of the specified column in the dataset.
        Args:
            column (str): The column name to plot.
        """
        assert self.loop_set, "plot_histogram is only available for loop set datasets."
        all_resistance = np.concatenate(
            [loop.resistance for loop in self.all_loops])
        all_reactance = np.concatenate(
            [loop.reactance for loop in self.all_loops])
        all_volume = np.concatenate([loop.volume for loop in self.all_loops])
        all_flow = np.concatenate([loop.flow for loop in self.all_loops])
        print(
            f"Resistance: {all_resistance.shape}, Reactance: {all_reactance.shape}, Volume: {all_volume.shape}, Flow: {all_flow.shape}")

        # Create 2x2 subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Resistance", "Reactance", "Volume", "Flow")
        )

        # Add histograms
        fig.add_trace(go.Histogram(x=all_resistance, nbinsx=30,
                      name="Resistance"), row=1, col=1)
        fig.add_trace(go.Histogram(x=all_reactance, nbinsx=30,
                      name="Reactance"), row=1, col=2)
        fig.add_trace(go.Histogram(x=all_volume, nbinsx=30,
                      name="Volume"), row=2, col=1)
        fig.add_trace(go.Histogram(
            x=all_flow, nbinsx=30, name="Flow"), row=2, col=2)

        fig.update_layout(
            title_text="Histograms of Loop Parameters",
            showlegend=False,
            height=700, width=900
        )

        fig.update_xaxes(title_text="Resistance", row=1, col=1)
        fig.update_xaxes(title_text="Reactance", row=1, col=2)
        fig.update_xaxes(title_text="Volume", row=2, col=1)
        fig.update_xaxes(title_text="Flow", row=2, col=2)

        fig.update_yaxes(title_text="Frequency")
        fig.show()

    @staticmethod
    def create_loops(df: pd.DataFrame,
                     labels: pd.Series) -> tuple[List[List[Loop]], NDArray[np.str_]]:
        """
        Create lists of loops of the same patient from the DataFrame.
        Each loop is defined by a unique patient_id and loop_id.

        Args:
            df (pd.DataFrame): DataFrame containing the loop data with columns:
                - 'id': Patient ID
                - 'index': Loop ID
                - 'volume': Volume data
                - 'flow': Flow data
                - 'resistance': Resistance data
                - 'reactance': Reactance data
            labels (pd.Series): Series containing the labels for each loop.
        Returns:
            list: A list of Loop objects of the same patient for each patient, created from the DataFrame.
            np.array: An array of labels corresponding to the patients.
        """
        sorted_idx = df.sort_values(by='id').index  # sort by patient ID
        df.sort_values(by='id', inplace=True)
        df.reset_index(drop=True, inplace=True)
        labels = labels.loc[sorted_idx].reset_index(drop=True)

        patient_groups = df.groupby('id').groups
        patient_labels = [labels.loc[patient_groups[id][0]]
                          for id in pd.unique(df['id'])]

        patients = [[
            Loop(
                row['id'],
                row['index'],
                [row.iloc[i] for i in range(3, 53)],
                [row.iloc[i] for i in range(53, 103)],
                [row.iloc[i] for i in range(103, 153)],
                [row.iloc[i] for i in range(153, 203)]) for _,
            row in group.iterrows()] for _, group in df.groupby('id')]
        return patients, np.array(patient_labels)

    @staticmethod
    def plot(canvas: torch.Tensor):
        canvas = canvas[0]
        assert canvas.ndim == 2 or canvas.ndim == 3
        if canvas.ndim == 3:
            canvas_np = canvas.permute(1, 2, 0).numpy()
        plt.imshow(canvas_np, cmap='gray')
        plt.axis("off")
        plt.show()


def init_dataset(params: Dict) -> tuple[DataLoader, DataLoader, DataLoader, IBDataset]:
    """
    Initializes the dataset and DataLoaders for training, testing and validation in ttv_split proportion in params.

    Args:
        params (dict):
            data_path (str | Path | list[str  |  Path]): Path to the dataset or list of paths.
            batch_size (int): Batch size for the DataLoaders.
            class_labels (Optional[str]): Class labels in order: healthy, COPD, ILD, asthma.
            If None, defaults to ["healthy", "COPD", "ILD", "asthma"].
            train_loops (bool): If True, train on loops, else on patient loops.
            seed (int): Random seed
            ttv_ratio (list[float]): List of ratios for train, test, validation split. Defaults to [0.7, 0.2, 0.1].

    Returns:
        tuple[DataLoader, DataLoader, DataLoader] : DataLoader for train (70%), validation(10%) and test (20%), and the full dataset to get its properties.
    """
    from IB_DL.transformations import build_transformations
    # The split is drawn from the global RNG, which callers seed with
    # params["seed"], so by default one seed fixes both the split and the
    # initialisation. With split_seed set, the split is drawn from split_seed
    # instead and the training seed is restored afterwards (Section VI-K).
    split_seed = params.get("split_seed")
    if isinstance(split_seed, str):
        # From the CLI: a None default makes click type the option as text.
        split_seed = None if split_seed in ("", "None") else int(split_seed)
        params["split_seed"] = split_seed
    if split_seed is not None:
        set_seed(split_seed)

    ib_dataset = None
    file_path: str | Path | list[str | Path] = ""
    if params["all_measurements"]:
        file_path = [os.path.join(params["data_dir"], i) for i in MEASURENT_FILES]
    else:
        file_path = os.path.join(params["data_dir"], "loops_filtered.csv")

    ib_dataset = IBDataset(
        file_path, transform=build_transformations(params),
        target_transform=None, class_labels=params["class_labels"],
        num_of_rows=params["num_rows"],
        loop_set=False
        if
        any(
            "MedianLoops" == trans["name"] or "ToMeasurements" == trans["name"]
            for trans in params["transformations"]) else True,
        test_aggregation=params["test_aggregation"])
    params["class_labels"] = ib_dataset.class_labels
    print(
        f"Dataset loaded: loops used: {ib_dataset.loop_set}; size: {len(ib_dataset)}")

    data_subsets = ib_dataset.split_data_subsets(
        ttv_ratio=params["ttv_ratio"])
    if split_seed is not None:
        # Sampling, augmentation and model initialisation follow the training
        # seed, not the split seed.
        set_seed(params["seed"])
    effective_split_seed = split_seed if split_seed is not None else params["seed"]

    # Optionally dump the exact patient-level split (train/test/val patient IDs).
    if params.get("save_patient_split", False):
        out_dir = params.get("output_dir", ".")
        split_path = os.path.join(
            out_dir, f"patient_split_seed{effective_split_seed}.json")
        ib_dataset.save_patient_split(
            split_path,
            extra={"seed": params["seed"],
                   "split_seed": effective_split_seed,
                   "ttv_ratio": list(params["ttv_ratio"]),
                   "loop_set": ib_dataset.loop_set})
        print(f"Saved patient split to {split_path}")

    data_loaders = []
    # In loop mode the subset indices address all_annotations, in patient mode
    # (MedianLoops / ToMeasurements) they address annotations.
    train_base_dataset = data_subsets[0].dataset
    train_targets_source = (train_base_dataset.all_annotations  # type: ignore
                            if train_base_dataset.loop_set  # type: ignore
                            else train_base_dataset.annotations)  # type: ignore
    subset_targets = [str(train_targets_source[idx])
                      for idx in data_subsets[0].indices]
    unique_classes = sorted(list(set(subset_targets)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
    int_targets = [class_to_idx[t] for t in subset_targets]
    class_counts = np.bincount(int_targets)
    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    sample_weights = [class_weights[class_to_idx[t]] for t in subset_targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights=sample_weights,  # type: ignore
                                    num_samples=len(sample_weights),
                                    replacement=True)
    g = torch.Generator()
    g.manual_seed(params["seed"])

    data_loaders.append(DataLoader(data_subsets[0],
                                   batch_size=params["batch_size"],
                                   shuffle=False,
                                   sampler=sampler,
                                   num_workers=params["num_workers"],
                                   persistent_workers=params["persistent_workers"],
                                   prefetch_factor=params["prefetch_factor"],
                                   pin_memory=False,
                                   generator=g,
                                   worker_init_fn=seed_worker),
                        )
    for i in range(1, 3):
        data_loaders.append(DataLoader(data_subsets[i],
                                       batch_size=params["batch_size"],
                                       shuffle=False,
                                       num_workers=params["num_workers"],
                                       persistent_workers=params["persistent_workers"],
                                       prefetch_factor=params["prefetch_factor"],
                                       pin_memory=False,
                                       generator=g,
                                       worker_init_fn=seed_worker),
                            )
    print(
        f"Train size: {len(data_subsets[0])}, test size: {len(data_subsets[2])}, validation size: {len(data_subsets[1])}")
    return data_loaders[0], data_loaders[2], data_loaders[1], ib_dataset
