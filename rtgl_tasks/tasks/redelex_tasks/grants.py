"""Collection of pre-defined RTGL tasks on ReDeLEx Grants dataset."""

from relbench import load_dataset
from relbench.base import TaskType

from rtgl_tasks.base import RTGLStatTask, get_redelex_path

########### Static Tasks ###########

class AwardsInstitutionStatTask(RTGLStatTask):
    """Predict the institution connected to each award."""

    dataset = load_dataset(get_redelex_path("ctu-grants"))
    entity_table = "awards"
    task_type = TaskType.MULTICLASS_CLASSIFICATION

    rtgl_query = """
        PREDICT institution_awards.FK_institution_name_zipcode
        FOR EACH awards.*;
    """


class CountInstitutionAwardsStatTask(RTGLStatTask):
    """Predict the number of distinct awards per institution."""

    dataset = load_dataset(get_redelex_path("ctu-grants"))
    entity_table = "institution"
    task_type = TaskType.REGRESSION

    rtgl_query = """
        PREDICT COUNT_DISTINCT(institution_awards.*)
        FOR EACH institution.*;
    """


class OrganizationAwardsAmountStatTask(RTGLStatTask):
    """Predict total awarded amount for each organization."""

    dataset = load_dataset(get_redelex_path("ctu-grants"))
    entity_table = "organization"
    task_type = TaskType.REGRESSION

    rtgl_query = """
        PREDICT SUM(awards.award_amount)
        FOR EACH organization.*;
    """
