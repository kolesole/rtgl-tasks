"""Collection of pre-defined RTGL tasks on RelBench ArXiv dataset."""

import pandas as pd
from relbench import load_dataset
from relbench.base import TaskType

from rtgl_tasks.base import RTGLTmpTask

########### Temporal Tasks ###########

class PaperCitationTmpTask(RTGLTmpTask):
    """Predict if a paper gets cited in the next 6 months."""

    dataset = load_dataset("rel-arxiv")
    entity_table = "papers"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//2)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        WITH citations_papers AS (
            citations.References_Paper_ID->papers.Paper_ID
        )
        PREDICT COUNT(citations_papers.*, 0, 182, DAYS) != 0
        FOR EACH papers.*;
    """


class PaperCitationTmpTaskInjection(RTGLTmpTask):
    """Predict if a paper gets cited in the next 6 months."""

    dataset = load_dataset("rel-arxiv")
    entity_table = "papers"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//2)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT COUNT(
            [
                SELECT
                    *
                FROM
                    citations
            ]{citations_papers}
            {}
            {References_Paper_ID->papers}
            {}
            {Submission_Date}.*, 0, 182, DAYS) != 0
        FOR EACH papers.*;
    """
