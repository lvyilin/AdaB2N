from functools import partial
from math import inf
from torch.nn import BatchNorm2d, Parameter
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Any


class AdaB2N(BatchNorm2d):
    def __init__(
        self,
        target: BatchNorm2d,
        num_tasks: int,
        num_classes_per_task: int,
        kappa: float = 1.0,
        init_weight: float = 0.0,
    ) -> None:
        super().__init__(
            target.num_features,
            target.eps,
            target.momentum,
            target.affine,
            target.track_running_stats,
        )
        assert self.momentum is not None
        assert self.track_running_stats
        assert 0.0 <= kappa <= 1.0

        self.load_state_dict(target.state_dict())

        self.kappa = kappa
        self.last_eta = None
        self.num_classes_per_task = num_classes_per_task
        self.num_tasks = num_tasks
        self.init_weight = init_weight
        self.register_buffer("cur_tasks", torch.tensor(0, dtype=torch.long))
        self.cur_tasks: Optional[Tensor]

        self.init_task_weight()

        self.sample_task_indices = None
        self.sample_task_counts = None
        self.task_counts_extended = None

    def init_task_weight(self):
        self.task_weight = Parameter(
            torch.full((self.num_tasks,), fill_value=self.init_weight)
        )

    def set_counts(self, sample_task_indices, sample_task_counts, task_counts_extended):
        self.sample_task_indices = sample_task_indices
        self.sample_task_counts = sample_task_counts
        self.task_counts_extended = task_counts_extended

    def end_task(self):
        self.cur_tasks.add_(1)

    def get_eta(self) -> float:
        if self.kappa == 0.0:
            return 1.0 / float(self.num_batches_tracked)
        if self.kappa == 1.0:
            return self.momentum
        if self.last_eta is None:
            self.last_eta = self.momentum**self.kappa
            return self.last_eta
        eta = self.last_eta / (self.last_eta + (1 - self.momentum) ** self.kappa)
        self.last_eta = eta
        return eta

    def training_forward(self, input: Tensor) -> Tensor:
        self.num_batches_tracked.add_(1)
        eta = self.get_eta()

        if self.sample_task_indices is not None and self.cur_tasks > 0:
            concentration = (
                self.task_weight[: self.cur_tasks + 1].exp() + self.task_counts_extended
            )
            task_weights = concentration / concentration.sum()
            sample_weights = (
                task_weights[self.sample_task_indices] / self.sample_task_counts
            )

            batch_mean = input.mean([2, 3]).t().matmul(sample_weights).view(1, -1, 1, 1)
            batch_var = (
                (input - batch_mean)
                .square()
                .mean([2, 3])
                .t()
                .matmul(sample_weights)
                .view(1, -1, 1, 1)
            )
        else:
            batch_var, batch_mean = torch.var_mean(
                input, [0, 2, 3], correction=0, keepdim=True
            )

        output = (input - batch_mean) / torch.sqrt(batch_var + self.eps)
        if self.affine:
            output = self.weight.view(1, -1, 1, 1) * output + self.bias.view(
                1, -1, 1, 1
            )

        self.running_mean.add_(
            batch_mean.detach().squeeze() - self.running_mean, alpha=eta
        )
        self.running_var.mul_(1 - eta).add_(batch_var.detach().squeeze(), alpha=eta)
        self.loss = F.mse_loss(batch_mean.squeeze(), self.running_mean) + F.mse_loss(
            batch_var.squeeze(), self.running_var
        )

        return output

    def eval_forward(self, input: Tensor) -> Tensor:
        # output = (input - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
        #     self.running_var.view(1, -1, 1, 1) + self.eps
        # )
        # if self.affine:
        #     output = self.weight.view(1, -1, 1, 1) * output + self.bias.view(
        #         1, -1, 1, 1
        #     )
        # return output

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            0.0,
            self.eps,
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        if self.training:
            output = self.training_forward(input)
        else:
            output = self.eval_forward(input)
        return output
