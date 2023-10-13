# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

best_args = {
    "seq-cifar10": {
        "sgd": {-1: {"lr": 0.1, "batch_size": 32, "n_epochs": 50}},
        "derpp": {
            500: {
                "lr": 0.03,
                "minibatch_size": 32,
                "alpha": 0.2,
                "beta": 0.5,
                "batch_size": 32,
                "n_epochs": 50,
                
            },
            2000: {
                "lr": 0.03,
                "minibatch_size": 32,
                "alpha": 0.1,
                "beta": 1.0,
                "batch_size": 32,
                "n_epochs": 50,
            },
        },
        "er_ace": {
            500: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "n_epochs": 50},
            2000: {"lr": 0.03, "minibatch_size": 32, "batch_size": 32, "n_epochs": 50},
        },
    },
    "seq-cifar100": {
        "sgd": {-1: {"lr": 0.03, "optim_mom": 0, "optim_wd": 0}},
        "er_ace": {
            2000: {"lr": 0.03, "optim_mom": 0, "optim_wd": 0},
            5000: {"lr": 0.03, "optim_mom": 0, "optim_wd": 0},
        },
        "derpp": {
            2000: {
                "lr": 0.03,
                "optim_mom": 0,
                "optim_wd": 0,
                "alpha": 0.1,
                "beta": 0.5,
            },
            5000: {
                "lr": 0.03,
                "optim_mom": 0,
                "optim_wd": 0,
                "alpha": 0.1,
                "beta": 0.5,
            },
        },
    },
    "seq-miniimg": {
        "derpp": {
            2000: {
                "lr": 0.1,
                "minibatch_size": 32,
                "alpha": 0.3,
                "beta": 0.8,
                "batch_size": 32,
                "n_epochs": 50,
            },
            5000: {
                "lr": 0.1,
                "minibatch_size": 32,
                "alpha": 0.3,
                "beta": 0.8,
                "batch_size": 32,
                "n_epochs": 50,
            },
        },
        "er_ace": {
            2000: {
                "lr": 0.1,
                "minibatch_size": 32,
                "batch_size": 32,
                "n_epochs": 50,
            },
            5000: {"lr": 0.1, "minibatch_size": 32, "batch_size": 32, "n_epochs": 50},
        },
    },
}
