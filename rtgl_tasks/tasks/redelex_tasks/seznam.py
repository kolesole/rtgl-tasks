"""Collection of pre-defined RTGL tasks on ReDeLEx Seznam dataset."""

import pandas as pd
from relbench import load_dataset
from relbench.base import TaskType

from rtgl_tasks.base import RTGLTmpTask, get_redelex_path

########### Temporal Tasks ###########

class ClientOutOfWalletTmpTask(RTGLTmpTask):
    """Predict whether a client will spend outside wallet in the next 30 days."""

    dataset = load_dataset(get_redelex_path("ctu-seznam"))
    entity_table = "client"
    task_type = TaskType.BINARY_CLASSIFICATION
    num_eval_timestamps = 3

    timedelta = pd.Timedelta(days=30)
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")

    rtgl_query = """
        PREDICT COUNT(probehnuto_mimo_penezenku.*, 0, 30, DAYS) != 0
        FOR EACH client.*
        WHERE COUNT(probehnuto.*, -inf, 0, DAYS) != 0
           OR COUNT(dobito.*, -inf, 0, DAYS) != 0
           OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, DAYS) != 0
        ASSUMING COUNT(probehnuto.*, 0, 30, DAYS) == 0;
    """


class ClientServisTmpTask(RTGLTmpTask):
    """Predict services a client will use in the next 30 days."""

    dataset = load_dataset(get_redelex_path("ctu-seznam"))
    entity_table = "client"
    task_type = TaskType.MULTILABEL_CLASSIFICATION
    num_eval_timestamps = 3

    timedelta = pd.Timedelta(days=30)
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")

    rtgl_query = """
        PREDICT LIST_DISTINCT(probehnuto.sluzba, 0, 30, DAYS)
        FOR EACH client.*
        WHERE COUNT(probehnuto.*, -inf, 0, DAYS) != 0
           OR COUNT(dobito.*, -inf, 0, DAYS) != 0
           OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, DAYS) != 0;
    """


class ClientFirstServisTmpTask(RTGLTmpTask):
    """Predict the first sercis a client will use in the next 30 days."""

    dataset = load_dataset(get_redelex_path("ctu-seznam"))
    entity_table = "client"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
    num_eval_timestamps = 3

    timedelta = pd.Timedelta(days=30)
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")

    rtgl_query = """
        PREDICT FIRST(probehnuto.sluzba, 0, 30, DAYS)
        FOR EACH client.*
        WHERE COUNT(probehnuto.*, -inf, 0, DAYS) != 0
           OR COUNT(dobito.*, -inf, 0, DAYS) != 0
           OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, DAYS) != 0;
    """


class ClientSpendingTmpTask(RTGLTmpTask):
    """Predict client spending amount in the next 30 days."""

    dataset = load_dataset(get_redelex_path("ctu-seznam"))
    entity_table = "client"
    task_type = TaskType.REGRESSION
    num_eval_timestamps = 3

    timedelta = pd.Timedelta(days=30)
    val_timestamp = pd.Timestamp("2015-03-01")
    test_timestamp = pd.Timestamp("2015-07-01")

    rtgl_query = """
        PREDICT SUM(probehnuto.kc_proklikano, 0, 30, DAYS)
        FOR EACH client.*
        WHERE COUNT(probehnuto.*, -inf, 0, DAYS) != 0
           OR COUNT(dobito.*, -inf, 0, DAYS) != 0
           OR COUNT(probehnuto_mimo_penezenku.*, -inf, 0, DAYS) != 0;
    """
