from functools import partial
import torch.nn as nn


class InstanceNorm2d(nn.InstanceNorm2d):
    def __init__(
        self,
        target,
    ) -> None:
        super().__init__(
            target.num_features,
        )
