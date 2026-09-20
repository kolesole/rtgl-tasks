"""Collection of pre-defined RTGL tasks on ReDeLEx Sfscores dataset."""

import pandas as pd
from relbench import load_dataset
from relbench.base import TaskType

from rtgl_tasks.base import RTGLTmpTask, get_redelex_path

########### Temporal Tasks ###########

class BusinessesScoresTmpTask(RTGLTmpTask):
    """Predict maximum future scores for each business in the next 182 days."""

    dataset = load_dataset(get_redelex_path("ctu-sfscores"))
    entity_table = "businesses"
    task_type = TaskType.MULTICLASS_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//2)
    val_timestamp = pd.Timestamp("2015-12-17")
    test_timestamp = pd.Timestamp("2016-06-16")

    rtgl_query = """
        PREDICT MAX(inspections.score, 0, 182, DAYS)
        FOR EACH businesses.*;
    """
