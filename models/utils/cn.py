from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F


class _CN(_BatchNorm):
    def __init__(self, target):
        super(_CN, self).__init__(
            target.num_features,
            target.eps,
            target.momentum,
            target.affine,
            target.track_running_stats,
        )
        self.load_state_dict(target.state_dict())

        self.N = target.num_features
        self.setG()

    def setG(self):
        pass

    def _check_input_dim(self, input):
        pass

    def forward(self, input):
        # self._check_input_dim(input)
        out_gn = F.group_norm(input, self.G, None, None, self.eps)
        out = F.batch_norm(
            out_gn,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )
        return out


class CN32(_CN):
    def setG(self):
        self.G = 32
