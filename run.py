import argparse
import itertools
import os
import queue
from time import sleep
from utils.magic import Task, wait_gpu


def get_buffers(dataset):
    if dataset == "seq-cifar10":
        return [500, 2000]
    elif dataset == "seq-cifar100":
        return [2000, 5000]
    elif dataset == "seq-miniimg":
        return [2000, 5000]
    raise NotImplementedError


def get_run(
    args,
    seed,
    dataset,
    model,
    buf,
    method,
    no_affine,
    kappa=None,
    lambd=None,
    ada_t0=None,
):
    project_name = f"{args.wandb_project}_{dataset}"
    if method == "AdaB2N":
        method_name = f"{method}_kappa{kappa}_lambd{lambd}_ada{ada_t0}"
    else:
        method_name = method

    cmd = f"""
    {args.interpreter} ./utils/main.py \
    --load_best_args \
    --seed {seed} \
    --dataset {dataset} \
    --model {model} \
    --wandb_project {project_name} \
    --wandb_name {model}{buf}_{method_name}\
    --buffer_size {buf} \
    --epochs {args.epochs} \
    --nl {method} \
    --buffer_mode {args.buffer_mode}
    --bs {args.bs}
    """
    if method == "AdaB2N":
        cmd += f" --kappa {kappa} --lambd {lambd} --ada_t0 {ada_t0}"
    if no_affine:
        cmd += " --no_affine"
    if args.nowandb:
        cmd += " --nowand 1"
    cmd = cmd.split()
    run_name = f"{dataset}_ep{args.epochs}_{model}_buf{buf}_{method_name}_seed{seed}_{args.buffer_mode}"
    run = Task(run_name, cmd, args.logdir)
    return run


def main(args):
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    todo_queue = queue.Queue()
    running_list = list()
    print("=> preparing runs")

    for seed, dataset, method, model in itertools.product(
        range(args.num_seeds),
        args.datasets,
        args.methods,
        args.models,
    ):
        no_affine = True if method == "CN" else False
        # This is consistent with the official CN implementation, which does not include affine parameters in the optimizer.
        # Refer to https://github.com/phquang/Continual-Normalization/blob/main/mammoth/models/utils/continual_model.py

        for buf in get_buffers(dataset):
            if method != "AdaB2N":
                run = get_run(args, seed, dataset, model, buf, method, no_affine)
                todo_queue.put(run)
                continue
            kappa_range = [0.1, 0.4, 0.7, 1.0]
            lambd_range = (
                [0.01, 0.1, 1.0, 10.0]
                if dataset != "seq-miniimg"
                else [0.00001, 0.0001, 0.001, 0.01]
            )
            ada_t0_range = [0, 1]
            for kappa, lambd, ada_t0 in itertools.product(
                kappa_range, lambd_range, ada_t0_range
            ):
                run = get_run(
                    args,
                    seed,
                    dataset,
                    model,
                    buf,
                    method,
                    no_affine,
                    kappa,
                    lambd,
                    ada_t0,
                )
                todo_queue.put(run)

    print([t.name for t in todo_queue.queue])
    print(f"=> {todo_queue.qsize()} tasks will be run. Confirm? (y/n)")
    if input().lower().strip() != "y":
        return
    gpus = []
    while not todo_queue.empty():
        task = todo_queue.get()
        gpus = wait_gpu(
            num=1,
            usage_threshold=(
                args.usage_threshold if "miniimg" not in task.name else 0.1
            ),
            waitsecs=args.waitsecs,
            last_gpus=gpus,
        )
        task.start(gpus)
        print("=> run task {}".format(task))
        running_list.append(task)
        print("=> cold down {} seconds".format(args.coldsecs))
        sleep(args.coldsecs)

    print("=> all tasks submitted, waiting for finish...")
    for task in running_list:
        duration = task.wait()
        print("=> finish task '{}' in {}".format(task.name, duration))
    print("=> all tasks finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpreter", type=str, default="python", help="Interpreter location")
    parser.add_argument("--coldsecs", type=int, default=15, help="Seconds to cool down after starting a run")
    parser.add_argument("--waitsecs", type=int, default=25, help="Seconds to sleep while waiting for an idle GPU")
    parser.add_argument("--usage_threshold", type=float, default=0.3, help="Threshold for determining idle GPUs")
    parser.add_argument("--logdir", type=str, default="data/logs", help="The save path for the command line output of each run")
    parser.add_argument("--models", nargs="+", type=str, default=["derpp", "er_ace"], help="Continual learning models to run")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--bs", type=int, default=10, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["seq-cifar10", "seq-cifar100", "seq-miniimg"],
        help="Datasets to run"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["BN", "LN", "IN", "GN", "CN", "AdaB2N"],
        help="Normalization layers to replace BN"
    )
    parser.add_argument(
        "--buffer_mode", type=str, default="reservoir", choices=["reservoir", "ring"],
        help="Replay buffer mode"
    )
    parser.add_argument("--nowandb", action="store_true", help="Inhibit wandb logging")
    parser.add_argument(
        "--wandb_project", type=str, default="CL", help="Wandb project name"
    )

    args = parser.parse_args()

    main(args)
