"""Base RTGL task class."""

from abc import ABC, abstractmethod
from functools import cache

import numpy as np
from relbench.base import Dataset, TaskType
from rtgl.base import Table
from rtgl.converter import Converter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def get_redelex_path(dataset: str) -> str:
    r"""Wrap the dataset name with the redelex repo.

    Args:
        dataset (str): Dataset name.

    Returns:
        out (str): Dataset name with prefix.
    """
    return f"stanford-star/redelex/{dataset}"


class RTGLBaseTask(ABC):
    r"""Base RTGL task class with share attributes and methods.

    Attributes:
        converter (Converter): Converter to convert the RTGL query to a task table (Default=None).
        dataset (Dataset): Dataset to get the database for the task.
        rtgl_query (str): RTGL query to convert to a task table.
        task_type (TaskType): Task type of the task.
        entity_table (str): Name of the entity table in the database.
        entity_col (str): Name of the entity column in the task table (Default="fk").
        target_col (str): Name of the target column in the task table (Default="label").
    """
    # to be set by subclasses
    converter: Converter=None
    dataset: Dataset
    rtgl_query: str
    task_type: TaskType
    entity_table: str
    # for RECOMMENDATION tasks
    dst_table: str=None
    # same for all tasks
    entity_col: str="fk"
    target_col: str="label"

    def get_table(self, split: str, hide_labels: bool=False) -> Table:
        r"""Get the task table for the given split.

        Args:
            split (str): Split to get the task table for ("train"/"val"/"test").
            hide_labels (bool): Whether to hide the labels in the task table (Default=False).

        Returns:
            out (Table): Task table for the given split.
        """
        table = self._get_table(split)

        if hide_labels:
            table.df.drop(columns=["label"], inplace=True)

        return table

    def compute_metrics(self, logits: np.ndarray, labels: np.ndarray) -> dict:
        r"""Compute the metrics for the given logits and labels.

        Args:
            logits (np.ndarray): Logits predicted by the model.
            labels (np.ndarray): True labels.

        Returns:
            out (dict): Dictionary of metric names and their values.
        """
        match self.task_type:
            case TaskType.REGRESSION:
                return {
                    "mae": mean_absolute_error(labels, logits),
                    "mse": mean_squared_error(labels, logits),
                    "r2": r2_score(labels, logits)
                }
            case TaskType.BINARY_CLASSIFICATION:
                return {
                    "accuracy": accuracy_score(labels, logits > 0.5),
                    "roc_auc": roc_auc_score(labels, logits),
                    "average_precision": average_precision_score(labels, logits),
                    "f1": f1_score(labels, logits >= 0.5)
                }
            case TaskType.MULTICLASS_CLASSIFICATION:
                return {
                    "accuracy": accuracy_score(labels, logits.argmax(axis=1)),
                    "macro_f1": f1_score(labels, logits.argmax(axis=1), average="macro"),
                    "micro_f1": f1_score(labels, logits.argmax(axis=1), average="micro")
                }
            case TaskType.MULTILABEL_CLASSIFICATION:
                return {
                    "auprc_macro": average_precision_score(labels, logits, average="macro"),
                    "auprc_micro": average_precision_score(labels, logits, average="micro"),
                    "f1_macro": f1_score(labels, logits > 0.5, average="macro"),
                    "f1_micro": f1_score(labels, logits > 0.5, average="micro")
                }
            case TaskType.RECOMMENDATION:
                return {
                    "roc_auc": roc_auc_score(labels, logits),
                    "average_precision": average_precision_score(labels, logits),
                    "f1": f1_score(labels, logits >= 0.5)
                }
            case _:
                pass

    @abstractmethod
    @cache
    def _get_table(self, split: str) -> Table:
        r"""Get the task table for the given split.

        Must be implemented by the child class.

        Args:
            split (str): Split to get the task table for ("train"/"val"/"test").

        Returns:
            out (Table): Task table for the given split.
        """
        pass
