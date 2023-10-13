from functools import partial
import torch.nn as nn


class LayerNorm(nn.LayerNorm):
    def __init__(
        self,
        target,
    ) -> None:
        assert hasattr(target, "_input_size")
        super().__init__([target.num_features, target._input_size[2], target._input_size[2]])
