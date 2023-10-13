# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import (
    add_management_args,
    add_experiment_args,
    add_rehearsal_args,
    ArgumentParser,
)
from utils.buffer import Buffer
from utils.ring_buffer import RingBuffer
from torch.nn import BatchNorm2d


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Continual learning via" " Dark Experience Replay++."
    )
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument("--alpha", type=float, required=True, help="Penalty weight.")
    parser.add_argument("--beta", type=float, required=True, help="Penalty weight.")
    return parser


class DerppAda(ContinualModel):
    NAME = "derpp_ada"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(DerppAda, self).__init__(backbone, loss, args, transform)
        self.n_tasks = get_dataset(args).N_TASKS
        self.n_classes_per_task = get_dataset(args).N_CLASSES_PER_TASK

        buffer_fn = RingBuffer if args.buffer_mode == "ring" else Buffer
        self.buffer = buffer_fn(
            self.args.buffer_size, self.device, n_tasks=self.n_tasks
        )
        self.task = 0

        self.norm_modules = [
            m for m in self.net.modules() if isinstance(m, BatchNorm2d)
        ]

    def set_counts(self, labels):
        task_indices, inverse_indices, task_counts = torch.unique(
            labels // self.n_classes_per_task,
            return_inverse=True,
            return_counts=True,
        )
        sample_task_indices = task_indices[inverse_indices]
        sample_task_counts = task_counts[inverse_indices]
        task_counts_extended = labels.new_zeros(
            self.task + 1
        )  # in case a task not in batch
        task_counts_extended[task_indices] = task_counts
        for m in self.norm_modules:
            m.set_counts(sample_task_indices, sample_task_counts, task_counts_extended)

    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs1, buf_labels1, buf_logits1 = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_inputs2, buf_labels2, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            all_inputs = torch.cat((inputs, buf_inputs1, buf_inputs2), dim=0)
            all_labels = torch.cat((labels, buf_labels1, buf_labels2), dim=0)
            self.set_counts(all_labels)

            all_outputs = self.net(all_inputs)
            outputs, buf_outputs1, buf_outputs2 = torch.split(
                all_outputs,
                [inputs.shape[0], buf_inputs1.shape[0], buf_inputs2.shape[0]],
            )
            if self.task >= self.args.ada_t0:
                loss_norm = (
                    torch.stack([m.loss for m in self.norm_modules]).sum()
                    * self.args.lambd
                )
            else:
                loss_norm = torch.tensor(0.0, device=inputs.device)

            loss = (
                self.loss(outputs, labels)
                + self.args.alpha * F.mse_loss(buf_outputs1, buf_logits1)
                + self.args.beta * self.loss(buf_outputs2, buf_labels2)
            )
            loss += loss_norm
        else:
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs, labels=labels, logits=outputs.data
        )

        return loss.item()

    def end_task(self, dataset):
        self.task += 1
        if self.args.buffer_mode == "ring":
            self.buffer.task_number += 1

        for m in self.norm_modules:
            m.end_task()
