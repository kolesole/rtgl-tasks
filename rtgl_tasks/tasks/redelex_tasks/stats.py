"""Collection of pre-defined RTGL tasks on ReDeLEx Stats dataset."""

import pandas as pd
from relbench import load_dataset
from relbench.base import TaskType

from rtgl_tasks.base import RTGLStatTask, RTGLTmpTask, get_redelex_path

########### Static Tasks ###########

class UserReputationStatTask(RTGLStatTask):
    """Predict reputation for each active user."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "users"
    task_type = TaskType.REGRESSION

    rtgl_query = """
        WITH posts_users AS (
            posts.FK_users_OwnerUserId->users.__PK__
        )
        PREDICT users.reputation
        FOR EACH users.*
        WHERE COUNT(votes.*) != 0
           OR COUNT(comments.*) != 0
           OR COUNT(posts_users.*) != 0;
    """


class PostTagsStatTask(RTGLStatTask):
    """Predict the set of tags associated with each post."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "posts"
    task_type = TaskType.MULTILABEL_CLASSIFICATION

    rtgl_query = """
        PREDICT LIST_DISTINCT(tags.*)
        FOR EACH posts.*;
    """


class UserBadgeStatTask(RTGLStatTask):
    """Predict the total number of badges for each user."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "users"
    task_type = TaskType.REGRESSION

    rtgl_query = """
        PREDICT COUNT(badges.*)
        FOR EACH users.*;
    """


class UserEngagementStatTask(RTGLStatTask):
    """Predict whether a user is active."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    rtgl_query = """
        WITH posts_users AS (
            posts.FK_users_OwnerUserId->users.__PK__
        )
        PREDICT COUNT(votes.*) != 0
             OR COUNT(posts_users.*) != 0
             OR COUNT(comments.*) != 0
        FOR EACH users.*;
    """


class PostVotesStatTask(RTGLStatTask):
    """Predict the number of upvotes for each valid question post."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "posts"
    task_type = TaskType.REGRESSION

    rtgl_query = """
        PREDICT COUNT_DISTINCT(votes.*
                         WHERE votes.votetypeid == 2)
        FOR EACH posts.* WHERE posts.PostTypeId == 1
                           AND posts.OwnerUserId IS NOT NULL
                           AND posts.OwnerUserId != -1;
    """


class UserPostCommentStatTask(RTGLStatTask):
    """Predict which posts each user comments on."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "users"
    task_type = TaskType.RECOMMENDATION
    dst_table = "comments"

    rtgl_query = """
        PREDICT LIST_DISTINCT(comments.FK_posts_PostId
                        WHERE posts.owneruserid IS NOT NULL
                          AND posts.owneruserid != -1)
        FOR EACH users.*;
    """


class PostPostRelatedStatTask(RTGLStatTask):
    """Predict post-to-post related links."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "posts"
    task_type = TaskType.RECOMMENDATION
    dst_table = "postLinks"

    rtgl_query = """
        WITH links_posts AS (
            postLinks.FK_posts_PostId->posts.__PK__
        )
        PREDICT LIST_DISTINCT(links_posts.FK_posts_PostId)
        FOR EACH posts.*;
    """

########### Temporal Tasks ###########

class UserBadgeTmpTask(RTGLTmpTask):
    """Predict whether a user earns any badge in the next 91 days."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "users"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    rtgl_query = """
        PREDICT COUNT(badges.*, 0, 91, DAYS) != 0
        FOR EACH users.*;
    """


class UserEngagementTmpTask(RTGLTmpTask):
    """Predict whether a user is active in the next 91 days."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "users"
    task_type = TaskType.BINARY_CLASSIFICATION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    rtgl_query = """
        WITH posts_users AS (
            posts.FK_users_OwnerUserId->users.__PK__
        )
        PREDICT COUNT(votes.*, 0, 91, DAYS) != 0
             OR COUNT(posts_users.*, 0, 91, DAYS) != 0
             OR COUNT(comments.*, 0, 91, DAYS) != 0
        FOR EACH users.*
        WHERE COUNT(votes.*, -inf, 0, DAYS) != 0
                 OR COUNT(posts_users.*, -inf, 0, DAYS) != 0
                 OR COUNT(comments.*, -inf, 0, DAYS) != 0;
    """


class PostVotesTmpTask(RTGLTmpTask):
    """Predict future upvote count for each valid question post."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "posts"
    task_type = TaskType.REGRESSION

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    rtgl_query = """
        PREDICT COUNT_DISTINCT(votes.*
                         WHERE votes.votetypeid == 2, 0, 91, DAYS)
        FOR EACH posts.* WHERE posts.PostTypeId == 1
                           AND posts.OwnerUserId IS NOT NULL
                           AND posts.OwnerUserId != -1;
    """


class UserPostCommentTmpTask(RTGLTmpTask):
    """Predict posts each user will comment on in the next 91 days."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "users"
    task_type = TaskType.RECOMMENDATION
    dst_table = "posts"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    rtgl_query = """
        PREDICT LIST_DISTINCT(comments.FK_posts_PostId
                        WHERE posts.owneruserid IS NOT NULL
                          AND posts.owneruserid != -1, 0, 91, DAYS)
        FOR EACH users.*;
    """


class PostPostRelatedTmpTask(RTGLTmpTask):
    """Predict related posts linked from each post in the next 91 days."""

    dataset = load_dataset(get_redelex_path("ctu-stats"))
    entity_table = "posts"
    task_type = TaskType.RECOMMENDATION
    dst_table = "posts"

    timedelta = pd.Timedelta(days=365//4)
    val_timestamp = pd.Timestamp("2014-03-01")
    test_timestamp = pd.Timestamp("2014-06-01")

    rtgl_query = """
        WITH links_posts AS (
            postLinks.FK_posts_PostId->posts.__PK__
        )
        PREDICT LIST_DISTINCT(links_posts.FK_posts_RelatedPostId, 0, 91, DAYS)
        FOR EACH posts.*;
    """
