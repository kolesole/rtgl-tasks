"""Collection of pre-defined RTGL tasks on RelBench Stack dataset."""

import pandas as pd
from relbench import load_dataset
from relbench.base import TaskType

from rtgl_tasks.base import RTGLTmpTask

########### Temporal Tasks ###########

class UserEngagementTmpTask(RTGLTmpTask):
    """For each user predict if a user will make any votes, posts, or comments in the next 3 months."""

    dataset = load_dataset("rel-stack")
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT COUNT(votes.*, 0, 91, DAYS) != 0
             OR COUNT(posts.*, 0, 91, DAYS) != 0
             OR COUNT(comments.*, 0, 91, DAYS) != 0
        FOR EACH users.*
        WHERE COUNT(votes.*, -inf, 0, DAYS) != 0
           OR COUNT(posts.*, -inf, 0, DAYS) != 0
           OR COUNT(comments.*, -inf, 0, DAYS) != 0;
    """


class UserBadgeTmpTask(RTGLTmpTask):
    """For each user predict if a user will receive a new badge in the next 3 months."""

    dataset = load_dataset("rel-stack")
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT COUNT(badges.*, 0, 91, DAYS) != 0
        FOR EACH users.*;
    """


class PostVotesTmpTask(RTGLTmpTask):
    """For each user post predict how many votes it will receive in the next 3 months."""

    dataset = load_dataset("rel-stack")
    entity_table = "posts"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT COUNT_DISTINCT(votes.* WHERE votes.votetypeid == 2, 0, 91, DAYS)
        FOR EACH posts.* WHERE posts.PostTypeId == 1
                           AND posts.OwnerUserId IS NOT NULL
                           AND posts.OwnerUserId != -1;
    """


class UserPostCommentTmpTask(RTGLTmpTask):
    """Predict a list of existing posts that a user will comment in the next two months."""

    dataset = load_dataset("rel-stack")
    entity_table = "users"
    task_type = TaskType.RECOMMENDATION
    dst_table = "posts"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        PREDICT LIST_DISTINCT(comments.PostId
                        WHERE posts.owneruserid IS NOT NULL
                          AND posts.owneruserid != -1, 0, 91, DAYS)
        FOR EACH users.*;
    """


class PostPostRelatedTmpTask(RTGLTmpTask):
    """Predict a list of existing posts that users will link a given post to in the next two months."""

    dataset = load_dataset("rel-stack")
    entity_table = "posts"
    task_type = TaskType.RECOMMENDATION
    dst_table = "posts"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = dataset.val_timestamp
    test_timestamp = dataset.test_timestamp

    rtgl_query = """
        WITH links_posts AS (
            postLinks.PostId->posts.Id
        )
        PREDICT LIST_DISTINCT(links_posts.RelatedPostId, 0, 91, DAYS)
        FOR EACH posts.*;
    """
