"""Collection of pre-defined RTGL tasks on RelBench datasets."""

from .amazon import (
    ItemChurnTmpTask,
    ItemLTVTmpTask,
    ItemLTVTmpTaskInjection,
    UserChurnTmpTask,
    UserItemPurchaseTmpTask,
    UserItemRateTmpTask,
    UserItemReviewTmpTaskInjection,
    UserLTVTmpTask,
)
from .arxiv import (
    PaperCitationTmpTask,
    PaperCitationTmpTaskInjection,
)
from .f1 import (
    DriverCircuitCompleteTmpTask,
    DriverCircuitCompleteTmpTaskInjection,
    DriverDNFTmpTask,
    DriverPositionTmpTask,
    DriverTop3TmpTask,
)
from .stack import (
    PostPostRelatedTmpTask,
    PostVotesTmpTask,
    UserBadgeTmpTask,
    UserEngagementTmpTask,
    UserPostCommentTmpTask,
)

__all__ = [
    "UserChurnTmpTask",
    "ItemChurnTmpTask",
    "UserLTVTmpTask",
    "UserItemPurchaseTmpTask",
    "UserItemRateTmpTask",
    "ItemLTVTmpTask",
    "ItemLTVTmpTaskInjection",
    "UserItemReviewTmpTaskInjection",

    "PaperCitationTmpTask",
    "PaperCitationTmpTaskInjection",

    "DriverDNFTmpTask",
    "DriverTop3TmpTask",
    "DriverPositionTmpTask",
    "DriverCircuitCompleteTmpTask",
    "DriverCircuitCompleteTmpTaskInjection",

    "UserEngagementTmpTask",
    "UserBadgeTmpTask",
    "PostVotesTmpTask",
    "UserPostCommentTmpTask",
    "PostPostRelatedTmpTask"
]
