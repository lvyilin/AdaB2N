# persistent_locals has been co-authored with Andrea Maffezzoli

import os
import subprocess
import sys
from datetime import datetime
from io import StringIO
from time import sleep
from typing import Callable

import pandas as pd


class persistent_locals:
    def __init__(self, func: Callable):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event == "return":
                self._locals = frame.f_locals.copy()

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


def get_free_gpu(num: int = None, usage_threshold=1.0, verbose=True, return_str=True):
    if (num is not None and num <= 0) or usage_threshold > 1.0 or usage_threshold < 0.0:
        raise ValueError

    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.free,memory.total"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode("utf8")),
        names=["memory.free", "memory.total"],
        skiprows=1,
    )
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    gpu_df["memory.total"] = gpu_df["memory.total"].map(
        lambda x: int(x.rstrip(" [MiB]"))
    )
    gpu_df = gpu_df.sort_values(by="memory.free", ascending=False)
    if verbose:
        print("GPU usage:\n{}".format(gpu_df))

    gpu_usages = 1 - gpu_df["memory.free"] / gpu_df["memory.total"]

    free_gpus = []
    for i in range(len(gpu_usages)):
        if gpu_usages.iloc[i] < usage_threshold:
            free_gpus.append(int(gpu_usages.index[i]))

    if num is not None:
        if len(gpu_df) < num:
            raise RuntimeError("Not enough GPU")
        free_gpus = free_gpus[:num]
    if verbose:
        for gpu in free_gpus:
            print("Returning GPU{}".format(gpu))

    if return_str:
        return ",".join(str(x) for x in free_gpus)
    return free_gpus


def wait_gpu(num, waitsecs, usage_threshold, reserve_gpus=None, last_gpus=None):
    if num <= 0:
        raise ValueError

    first_print = True
    retry = True
    while True:
        free_gpu_ids = get_free_gpu(
            usage_threshold=usage_threshold, verbose=False, return_str=False
        )
        if reserve_gpus is not None and len(reserve_gpus) > 0:
            free_gpu_ids = [i for i in free_gpu_ids if i not in reserve_gpus]
        if len(free_gpu_ids) < num:
            print("=> waiting GPU." if first_print else ".", end="")
            first_print = False
            sleep(waitsecs)
        else:
            print("=> found GPU {}".format(free_gpu_ids))
            new_gpus = free_gpu_ids[:num]
            if new_gpus == last_gpus and retry:
                sleep(waitsecs * 2)
                retry = False
                continue
            return new_gpus


class Task(object):
    def __init__(self, name, cmd, logdir="."):
        self.name = name
        self.cmd = cmd
        self.proc = None
        self.add_date = datetime.now()
        self.finish_date = None
        self.duration = None
        self.gpu = None
        self.filename = os.path.join(logdir, name + ".txt")

    def assign_gpu(self, gpu):
        self.gpu = gpu

    def start(self, gpu=None):
        self.gpu = gpu if gpu is not None else [0]
        self.proc = self.run_command(self.cmd, self.gpu, self.filename)

    def wait(self):
        self.proc.wait()
        self.finish_date = datetime.now()
        self.duration = self.finish_date - self.add_date
        return self.duration

    def is_stop(self):
        return self.proc is None or self.proc.poll() is not None

    def __repr__(self):
        return "{}::GPU{}::{}".format(self.name, self.gpu, " ".join(self.cmd))

    @staticmethod
    def run_command(cmd, gpu, filename):
        pipe = open(filename, "w")
        pipe.write(" ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            stdout=pipe,
            stderr=subprocess.STDOUT,
            env={
                "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu),
            },
            shell=False,
        )
        return proc
