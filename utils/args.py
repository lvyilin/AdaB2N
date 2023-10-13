# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_NAMES,
        help="Which dataset to perform experiments on.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name.", choices=get_all_models()
    )

    parser.add_argument("--lr", type=float, required=True, help="Learning rate.")

    parser.add_argument(
        "--optim_wd", type=float, default=0.0, help="optimizer weight decay."
    )
    parser.add_argument(
        "--optim_mom", type=float, default=0.0, help="optimizer momentum."
    )
    parser.add_argument(
        "--optim_nesterov", type=int, default=0, help="optimizer nesterov momentum."
    )

    parser.add_argument("--n_epochs", type=int, help="Batch size.")
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--distributed", type=str, default="no", choices=["no", "dp", "ddp"]
    )
    add_custom_args(parser)

def add_custom_args(parser: ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=-1, help="overwrites the #epochs")
    parser.add_argument(
        "--no_affine", action="store_true", help="disable affine parameters"
    )
    parser.add_argument(
        "--no_task_shuffle", action="store_true", help="disable task order shuffling"
    )
    parser.add_argument(
        "--bs", type=int, default=None, help="overwrites the batch size"
    )
    parser.add_argument(
        "--nl", type=str, default="BN", choices=["BN", "LN", "IN", "GN", "CN", "AdaB2N"]
    )
    parser.add_argument(
        "--ada_t0", type=int, default=0, help="The first task to enable ada loss"
    )
    parser.add_argument(
        "--lambd", type=float, default=None, help="hyperparameter lambda"
    )
    parser.add_argument(
        "--kappa", type=float, default=None, help="hyperparameter kappa"
    )
    parser.add_argument(
        "--buffer_mode", type=str, default="reservoir", choices=["reservoir", "ring"]
    )


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="The random seed.")
    parser.add_argument("--notes", type=str, default=None, help="Notes for this run.")

    parser.add_argument(
        "--non_verbose",
        default=0,
        choices=[0, 1],
        type=int,
        help="Make progress bars non verbose",
    )
    parser.add_argument(
        "--disable_log", default=0, choices=[0, 1], type=int, help="Enable csv logging"
    )

    parser.add_argument(
        "--validation",
        default=0,
        choices=[0, 1],
        type=int,
        help="Test on the validation set",
    )
    parser.add_argument(
        "--ignore_other_metrics",
        default=0,
        choices=[0, 1],
        type=int,
        help="disable additional metrics",
    )
    parser.add_argument(
        "--debug_mode",
        type=int,
        default=0,
        help="Run only a few forward steps per epoch",
    )
    parser.add_argument(
        "--nowand", default=0, choices=[0, 1], type=int, help="Inhibit wandb logging"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="regaz", help="Wandb entity"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="CL", help="Wandb project name"
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb group name")


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument(
        "--buffer_size", type=int, required=True, help="The size of the memory buffer."
    )
    parser.add_argument(
        "--minibatch_size", type=int, help="The batch size of the memory buffer."
    )


def add_aux_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used to load initial (pretrain) checkpoint
    :param parser: the parser instance
    """
    parser.add_argument(
        "--pre_epochs", type=int, required=False, help="pretrain_epochs."
    )
    parser.add_argument(
        "--datasetS",
        type=str,
        required=False,
        choices=["cifar100", "tinyimgR", "imagenet"],
    )
    parser.add_argument("--load_cp", type=str, default=None)
    parser.add_argument("--stop_after_prep", action="store_true")
