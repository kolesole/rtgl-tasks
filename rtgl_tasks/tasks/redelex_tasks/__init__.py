"""Collection of pre-defined RTGL tasks on CTU datasets."""

from .grants import (
    AwardsInstitutionStatTask,
    CountInstitutionAwardsStatTask,
    OrganizationAwardsAmountStatTask,
)
from .seznam import (
    ClientFirstServisTmpTask,
    ClientOutOfWalletTmpTask,
    ClientServisTmpTask,
    ClientSpendingTmpTask,
)
from .sfscores import BusinessesScoresTmpTask
from .stats import (
    PostPostRelatedStatTask,
    PostPostRelatedTmpTask,
    PostTagsStatTask,
    PostVotesStatTask,
    PostVotesTmpTask,
    UserBadgeStatTask,
    UserBadgeTmpTask,
    UserEngagementStatTask,
    UserEngagementTmpTask,
    UserPostCommentStatTask,
    UserPostCommentTmpTask,
    UserReputationStatTask,
)

__all__ = [
    "AwardsInstitutionStatTask",
    "CountInstitutionAwardsStatTask",
    "OrganizationAwardsAmountStatTask",

    "ClientOutOfWalletTmpTask",
    "ClientServisTmpTask",
    "ClientFirstServisTmpTask",
    "ClientSpendingTmpTask",

    "BusinessesScoresTmpTask",

    "UserReputationStatTask",
    "PostTagsStatTask",
    "UserBadgeStatTask",
    "UserEngagementStatTask",
    "PostVotesStatTask",
    "UserPostCommentStatTask",
    "PostPostRelatedStatTask",

    "UserBadgeTmpTask",
    "UserEngagementTmpTask",
    "PostVotesTmpTask",
    "UserPostCommentTmpTask",
    "PostPostRelatedTmpTask",
]
