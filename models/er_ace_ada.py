# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
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


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Continual learning via" " Experience Replay.")
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACEAda(ContinualModel):
    NAME = "er_ace_ada"
    COMPATIBILITY = ["class-il", "task-il"]

    def __init__(self, backbone, loss, args, transform):
        super(ErACEAda, self).__init__(backbone, loss, args, transform)
        self.n_tasks = get_dataset(args).N_TASKS
        self.n_classes_per_task = get_dataset(args).N_CLASSES_PER_TASK

        buffer_fn = RingBuffer if args.buffer_mode == "ring" else Buffer
        self.buffer = buffer_fn(
            self.args.buffer_size, self.device, n_tasks=self.n_tasks
        )
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = (
            get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        )
        self.task = 0

        self.norm_modules = [
            m for m in self.net.modules() if isinstance(m, torch.nn.BatchNorm2d)
        ]

    def end_task(self, dataset):
        self.task += 1
        if self.args.buffer_mode == "ring":
            self.buffer.task_number += 1

        for m in self.norm_modules:
            m.end_task()

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

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            all_inputs = torch.cat((inputs, buf_inputs), dim=0)
            all_labels = torch.cat((labels, buf_labels), dim=0)
            self.set_counts(all_labels)
            all_outputs = self.net(all_inputs)
            logits, buf_outputs = torch.split(
                all_outputs, [inputs.shape[0], buf_inputs.shape[0]]
            )
            if self.task >= self.args.ada_t0:
                loss_norm = (
                    torch.stack([m.loss for m in self.norm_modules]).sum()
                    * self.args.lambd
                )
            else:
                loss_norm = torch.tensor(0.0, device=inputs.device)
            loss_re = self.loss(buf_outputs, buf_labels)

            mask = torch.zeros_like(logits)
            mask[:, present] = 1

            if self.seen_so_far.max() < (self.num_classes - 1):
                mask[:, self.seen_so_far.max() :] = 1
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

            loss = self.loss(logits, labels) + loss_re

            loss += loss_norm

        else:
            logits = self.net(inputs)
            loss_re = 0.0
            loss = self.loss(logits, labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()
