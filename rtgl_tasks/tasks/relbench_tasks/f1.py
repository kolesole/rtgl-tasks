"""Collection of pre-defined RTGL tasks on RelBench F1 dataset."""

import pandas as pd
from relbench import load_dataset
from relbench.base import TaskType

from rtgl_tasks.base import RTGLTmpTask

########### Temporal Tasks ###########

class DriverDNFTmpTask(RTGLTmpTask):
    """For each driver predict the if they will DNF (did not finish) a race in the next 1 month."""

    dataset = load_dataset("rel-f1")
    entity_table = "drivers"
    task_type = TaskType.BINARY_CLASSIFICATION
    num_eval_timestamps = 40

    timedelta = pd.Timedelta(days=30)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT MAX(results.statusId, 0, 30, DAYS) != 1
        FOR EACH drivers.*
        WHERE COUNT(results.*, -365, 0, DAYS) != 0
        ASSUMING MAX(results.statusId, 0, 30, DAYS) IS NOT NULL;
    """


class DriverTop3TmpTask(RTGLTmpTask):
    """For each driver predict if they will qualify in the top-3 for a race in the next 1 month."""

    dataset = load_dataset("rel-f1")
    entity_table = "drivers"
    task_type = TaskType.BINARY_CLASSIFICATION
    num_eval_timestamps = 40

    timedelta = pd.Timedelta(days=30)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT MIN(qualifying.position, 0, 30, DAYS) <= 3
        FOR EACH drivers.*
        ASSUMING MIN(qualifying.position, 0, 30, DAYS) IS NOT NULL;
    """


class DriverPositionTmpTask(RTGLTmpTask):
    """Predict the average finishing position of each driver all races in the next 2 months."""

    dataset = load_dataset("rel-f1")
    entity_table = "drivers"
    task_type = TaskType.REGRESSION
    num_eval_timestamps = 40

    timedelta = pd.Timedelta(days=60)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT AVG(results.positionOrder, 0, 60, DAYS)
        FOR EACH drivers.*;
    """


class DriverCircuitCompleteTmpTask(RTGLTmpTask):
    """Predict on which circuits a driver will compete in the next 1 year."""

    dataset = load_dataset("rel-f1")
    entity_table = "drivers"
    task_type = TaskType.RECOMMENDATION
    dst_table = "circuits"

    timedelta = pd.Timedelta(days=365)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        WITH circuits_drivers AS (
            circuits.circuitId->races.circuitId:raceId->results.raceId:driverId->drivers.driverId
        )
        PREDICT LIST_DISTINCT(circuits_drivers.*, 0, 365, DAYS)
        FOR EACH drivers.*;
    """


class DriverCircuitCompleteTmpTaskInjection(RTGLTmpTask):
    """Predict on which circuits a driver will compete in the next 1 year."""

    dataset = load_dataset("rel-f1")
    entity_table = "drivers"
    task_type = TaskType.RECOMMENDATION
    dst_table = "circuits"

    timedelta = pd.Timedelta(days=365)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT LIST_DISTINCT(
            [
                SELECT
                    dr.driverId,
                    rac.circuitId,
                    rac.date
                FROM
                    drivers AS dr
                JOIN
                    results AS res
                ON
                    res.driverId = dr.driverId
                JOIN
                    races AS rac
                ON
                    rac.raceId = res.raceId
            ]{circuits_drivers}
            {}
            {driverId->drivers, circuitId->circuits}
            {}
            {date}.circuitId, 0, 365, DAYS)
        FOR EACH drivers.*;
    """
