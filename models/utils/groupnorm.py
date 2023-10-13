from functools import partial
from torch.nn import GroupNorm

class GN(GroupNorm):
    def __init__(
        self,
        target,
        num_groups
    ) -> None:
        super().__init__(
            num_groups,
            target.num_features,
        )

GN32 = partial(GN, num_groups=32)