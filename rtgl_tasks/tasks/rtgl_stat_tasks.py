"""Collection of pre-defined RTGL static tasks on ctu datasets."""

from relbench.base import TaskType
from relbench.datasets import get_dataset

from rtgl_tasks.base import RTGLStatTask

######### DATASET: сtu-stats #########

class StatsUserReputationStatTask(RTGLStatTask):
    """Predict reputation for each active user."""

    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.REGRESSION

    rtgl_query = """
          PREDICT users.reputation
          FOR EACH users.*
          WHERE COUNT(votes.*) != 0
             OR COUNT(comments.*) != 0
             OR COUNT(posts.*) != 0;
     """


class StatsPostTagsStatTask(RTGLStatTask):
    """Predict the set of tags associated with each post."""

    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "posts"
    task_type = TaskType.MULTILABEL_CLASSIFICATION

    rtgl_query = """
          PREDICT LIST_DISTINCT(tags.*)
          FOR EACH posts.*;
     """


class StatsUserBadgeStatTask(RTGLStatTask):
    """Predict the total number of badges for each user."""

    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.REGRESSION

    rtgl_query = """
          PREDICT COUNT(badges.*)
          FOR EACH users.*;
     """


class StatsUserEngagementStatTask(RTGLStatTask):
    """Predict whether a user is active."""

    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    rtgl_query = """
          PREDICT COUNT(votes.*) != 0
               OR COUNT(posts.*) != 0
               OR COUNT(comments.*) != 0
          FOR EACH users.*;
     """


class StatsPostVotesStatTask(RTGLStatTask):
    """Predict the number of upvotes for each valid question post."""

    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "posts"
    task_type = TaskType.REGRESSION

    rtgl_query = """
          PREDICT COUNT_DISTINCT(votes.*
               WHERE votes.votetypeid == 2)
          FOR EACH posts.* WHERE posts.PostTypeId == 1
                             AND posts.OwnerUserId IS NOT NULL
                             AND posts.OwnerUserId != -1;
      """


class StatsUserPostCommentStatTask(RTGLStatTask):
    """Predict which posts each user comments on."""

    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "users"
    task_type = TaskType.LINK_PREDICTION
    dst_table = "comments"

    rtgl_query = """
          PREDICT LIST_DISTINCT(comments.FK_posts_PostId
               WHERE posts.owneruserid IS NOT NULL
                 AND posts.owneruserid != -1)
          FOR EACH users.*;
     """


class StatsPostPostRelatedStatTask(RTGLStatTask):
    """Predict post-to-post related links."""

    dataset = get_dataset("ctu-stats", download=False)
    entity_table = "posts"
    task_type = TaskType.LINK_PREDICTION
    dst_table = "postLinks"

    rtgl_query = """
          PREDICT LIST_DISTINCT(postLinks.FK_posts_PostId)
          FOR EACH posts.*;
     """

######### DATASET: сtu-grants #########

class GrantsAwardsInstitutionStatTask(RTGLStatTask):
    """Predict the institution connected to each award."""

    dataset = get_dataset("ctu-grants", download=False)
    entity_table = "awards"
    task_type = TaskType.MULTICLASS_CLASSIFICATION

    rtgl_query = """
          PREDICT institution_awards.FK_institution_name_zipcode
          FOR EACH awards.*;
     """


class GrantsCountInstitutionAwardsStatTask(RTGLStatTask):
    """Predict the number of distinct awards per institution."""

    dataset = get_dataset("ctu-grants", download=False)
    entity_table = "institution"
    task_type = TaskType.REGRESSION

    rtgl_query = """
          PREDICT COUNT_DISTINCT(institution_awards.*)
          FOR EACH institution.*;
     """


class GrantsOrganizationAwardsAmountStatTask(RTGLStatTask):
    """Predict total awarded amount for each organization."""

    dataset = get_dataset("ctu-grants", download=False)
    entity_table = "organization"
    task_type = TaskType.REGRESSION

    rtgl_query = """
          PREDICT SUM(awards.award_amount)
          FOR EACH organization.*;
     """
