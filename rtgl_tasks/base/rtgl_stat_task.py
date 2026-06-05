"""Static RTGL task class."""

from copy import deepcopy

import numpy as np
from rtgl.base import Table
from rtgl.converter import SConverter

from .rtgl_base_task import RTGLBaseTask


class RTGLStatTask(RTGLBaseTask):
    r"""Static RTGL task class.

    This class is used for classic static tasks without time predictions.

    Atributes:
        target_table (Table): The attribute is used to store the task table without splitting (Default=None).
    """

    target_table: Table=None

    def _get_table(self, split: str) -> Table:
        r"""Get the task table for the given split.

        Args:
            split (str): Split to get the task table for ("train"/"val"/"test").
        """
        if self.converter is None:
            self.converter = SConverter(self.dataset.get_db(upto_test_timestamp=False))

        if self.target_table is None:
            self.target_table = self.converter.convert(self.rtgl_query, execute=True)

        table = deepcopy(self.target_table)

        random_state = np.random.RandomState(seed=42)

        train_df = table.df.sample(frac=0.8, random_state=random_state)
        if split == "train":
            table.df = train_df
        else:
            table.df = table.df.drop(train_df.index)
            val_df = table.df.sample(frac=0.5, random_state=random_state)
            if split == "val":
                table.df = val_df
            else:
                table.df = table.df.drop(val_df.index)

        return table
