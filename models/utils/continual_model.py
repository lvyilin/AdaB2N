# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import importlib
import sys
from argparse import Namespace
from contextlib import suppress
from typing import List

import torch
import torch.nn as nn
from torch.optim import SGD

from utils.conf import get_device
from utils.magic import persistent_locals

# from models.utils.plain_norm import PlainCN as CN32
# from models.utils.plain_norm import PlainBatchNorm2d as BatchNorm2d
# from models.utils.plain_norm import PlainGroupNorm as GN32
from models.utils.cn import CN32
from models.utils.groupnorm import GN32
from torch.nn import BatchNorm2d
from models.utils.layernorm import LayerNorm
from models.utils.instancenorm import InstanceNorm2d
from models.utils.adab2n import AdaB2N
from datasets import get_dataset

with suppress(ImportError):
    import wandb


def evaluate(nl, args):
    if nl == "CN":
        nl_fn = CN32
    elif nl == "GN":
        nl_fn = GN32
    elif nl == "LN":
        nl_fn = LayerNorm
    elif nl == "IN":
        nl_fn = InstanceNorm2d
    elif nl == "AdaB2N":
        num_classes_per_task = get_dataset(args).N_CLASSES_PER_TASK
        num_tasks = get_dataset(args).N_TASKS
        nl_fn = partial(
            AdaB2N,
            num_tasks=num_tasks,
            num_classes_per_task=num_classes_per_task,
            kappa=args.kappa,
        )
    elif nl == "BN":
        nl_fn = BatchNorm2d
    else:
        raise NotImplementedError
    return nl_fn


def disable_affine(m: nn.Module) -> None:
    if isinstance(m, nn.BatchNorm2d):
        if m.affine:
            m.affine = False
            m.weight = None
            m.bias = None


def save_input_size(m: nn.Module) -> None:
    def save_input_size_hook(module, args, output):
        assert len(output.shape) == 4
        assert output.size(2) == output.size(3)
        module._input_size = output.size()

    if isinstance(m, nn.BatchNorm2d):
        m.register_forward_hook(save_input_size_hook)


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """

    NAME: str
    COMPATIBILITY: List[str]

    def __init__(
        self,
        backbone: nn.Module,
        loss: nn.Module,
        args: Namespace,
        transform: nn.Module,
    ) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.device = get_device()
        self.nl_fn = evaluate(args.nl, args) if hasattr(args, "nl") else nn.BatchNorm2d
        if args.no_affine:
            self.net.apply(disable_affine)

        if args.nl == "LN":
            self.net.apply(save_input_size)
            status = self.net.training
            self.net.eval()
            with torch.no_grad():
                sz = get_dataset(args).get_input_size()
                self.net(torch.randn(1, 3, sz, sz))
            self.net.train(status)

        # if args.nl != "BN":
        print(f"=> replacing BN modules of the model to {args.nl}")
        self.replace_bn(self.net, "model", self.nl_fn)

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError(
                "Please specify the name and the compatibility of the model."
            )
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)

    def reset_classifier(self):
        self.net.classifier = torch.nn.Linear(
            self.net.classifier.in_features, self.net.num_classes
        ).to(self.device)
        self.opt = SGD(
            self.net.parameters(),
            lr=self.args.lr,
        )

    def replace_bn(self, module, name, nl):
        for name, sub_mod in module.named_children():
            if type(sub_mod) == nn.BatchNorm2d:
                new_bn = nl(sub_mod)
                setattr(module, name, new_bn)
            self.replace_bn(sub_mod, name, self.nl_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        if "wandb" in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def observe(
        self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor
    ) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:

            log_dict = {}
            for k, v in locals.items():
                if k.startswith("loss"):
                    if isinstance(v, torch.Tensor) and v.dim() == 0:
                        log_dict[k] = v.item()
                    else:
                        log_dict[k] = v
            wandb.log(log_dict)
