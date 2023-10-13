# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


class Derpp(ContinualModel):
    NAME = "derpp"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        buffer_fn = RingBuffer if args.buffer_mode == "ring" else Buffer
        self.buffer = buffer_fn(
            self.args.buffer_size, self.device, n_tasks=get_dataset(args).N_TASKS
        )

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs, labels=labels, logits=outputs.data
        )

        return loss.item()

    def end_task(self, dataset):
        if self.args.buffer_mode == "ring":
            self.buffer.task_number += 1
