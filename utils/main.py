# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import socket
import sys

import numpy  # needed (don't change it)

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + "/datasets")
sys.path.append(mammoth_path + "/backbone")
sys.path.append(mammoth_path + "/models")

import datetime
import uuid
from argparse import ArgumentParser
from time import time

import setproctitle
import torch

from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model
from utils.args import add_custom_args, add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.distributed import make_dp
from utils.magic import get_free_gpu
from utils.training import train


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description="mammoth", allow_abbrev=False)
    parser.add_argument(
        "--model", type=str, required=True, help="Model name.", choices=get_all_models()
    )
    parser.add_argument(
        "--load_best_args",
        action="store_true",
        help="Loads the best arguments for each method, " "dataset and memory buffer.",
    )
    add_custom_args(parser)

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module("models." + args.model)

    if args.load_best_args:
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            choices=DATASET_NAMES,
            help="Which dataset to perform experiments on.",
        )
        if hasattr(mod, "Buffer"):
            parser.add_argument(
                "--buffer_size",
                type=int,
                required=True,
                help="The size of the memory buffer.",
            )
        args = parser.parse_args()
        if args.model == "joint":
            best = best_args[args.dataset]["sgd"]
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, "Buffer"):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, "get_parser")
        parser = get_parser()
        to_parse = sys.argv[1:] + ["--" + k + "=" + str(v) for k, v in best.items()]
        to_parse.remove("--load_best_args")
        args = parser.parse_args(to_parse)
    else:
        get_parser = getattr(mod, "get_parser")
        parser = get_parser()
        args = parser.parse_args()
    if args.bs is not None:
        args.batch_size = args.bs
        args.minibatch_size = args.bs
    if args.epochs > 0:
        args.n_epochs = args.epochs
    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    _start_time = time()
    lecun_fix()
    if args is None:
        args = parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu(num=1, return_str=True)

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if (
        hasattr(importlib.import_module("models." + args.model), "Buffer")
        and args.minibatch_size is None
    ):
        args.minibatch_size = dataset.get_minibatch_size()
    if args.nl == "AdaB2N":
        args.model += "_ada"
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if args.distributed == "dp":
        model.net = make_dp(model.net)
        model.to("cuda:0")
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == "ddp":
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError("Distributed Data Parallel not supported yet.")

    if args.debug_mode:
        args.nowand = 1

    # set job name
    setproctitle.setproctitle(
        "{}_{}_{}".format(
            args.model, args.buffer_size if "buffer_size" in args else 0, args.dataset
        )
    )

    train(model, dataset, args)
    print(f"Run time in sec: {time() - _start_time}")


if __name__ == "__main__":
    main()
