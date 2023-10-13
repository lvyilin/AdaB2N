import torch
import torch.nn as nn
from torch import Tensor


class _PlainBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            batch_var, batch_mean = torch.var_mean(
                input, [0, 2, 3], correction=0, keepdim=True
            )

            output = (input - batch_mean) / torch.sqrt(batch_var + self.eps)
            if self.affine:
                output = self.weight.view(1, -1, 1, 1) * output + self.bias.view(
                    1, -1, 1, 1
                )

            self.running_mean.add_(
                batch_mean.detach().squeeze() - self.running_mean,
                alpha=exponential_average_factor,
            )
            n = input.numel() / input.size(1)  # update running_var with unbiased var
            self.running_var.mul_(1 - exponential_average_factor).add_(
                batch_var.detach().squeeze(),
                alpha=exponential_average_factor * n / (n - 1),
            )
            return output

        else:
            output = (input - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
                self.running_var.view(1, -1, 1, 1) + self.eps
            )
            if self.affine:
                output = self.weight.view(1, -1, 1, 1) * output + self.bias.view(
                    1, -1, 1, 1
                )
            return output


class PlainBatchNorm2d(_PlainBatchNorm2d):
    def __init__(self, target):
        super().__init__(
            target.num_features,
            target.eps,
            target.momentum,
            target.affine,
            target.track_running_stats,
        )


class _PlainGroupNorm(nn.GroupNorm):
    def forward(self, input: torch.Tensor):
        shape = input.shape
        batch_size = shape[0]
        assert self.num_channels == input.shape[1]

        input = input.view(batch_size, self.num_groups, -1)

        batch_var, batch_mean = torch.var_mean(input, [-1], correction=0, keepdim=True)

        output = (input - batch_mean) / torch.sqrt(batch_var + self.eps)

        if self.affine:
            output = output.view(batch_size, self.num_channels, -1)
            output = self.weight.view(1, -1, 1) * output + self.bias.view(1, -1, 1)

        return output.view(shape)


class PlainGroupNorm(_PlainGroupNorm):
    def __init__(self, target, G=32):
        super().__init__(
            G,
            target.num_features,
            target.eps,
        )


class PlainCN(nn.Module):
    def __init__(self, target, G=32):
        super().__init__()
        self.bn = _PlainBatchNorm2d(
            target.num_features,
            target.eps,
            target.momentum,
            target.affine,
            target.track_running_stats,
        )
        self.bn.load_state_dict(target.state_dict())
        self.gn = _PlainGroupNorm(G, target.num_features, target.eps, False)

    def forward(self, input: torch.Tensor):
        out_gn = self.gn(input)
        out = self.bn(out_gn)
        return out
