# Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation

This repository is the official implementation of *Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation* (NeurIPS 2023).

> Continual learning entails learning a sequence of tasks and balancing their knowledge appropriately. With limited access to old training samples, much of the current work in deep neural networks has focused on overcoming catastrophic forgetting of old tasks in gradient-based optimization. However, the normalization layers provide an exception, as they are updated interdependently by the gradient and statistics of currently observed training samples, which require specialized strategies to mitigate recency bias. In this work, we focus on the most popular Batch Normalization (BN) and provide an in-depth theoretical analysis of its sub-optimality in continual learning. Our analysis demonstrates the dilemma between balance and adaptation of BN statistics for incremental tasks, which potentially affects training stability and generalization. Targeting on these particular challenges, we propose Adaptive Balance of BN (AdaB$^2$N), which incorporates appropriately a Bayesian-based strategy to adapt task-wise contributions and a modified momentum to balance BN statistics, corresponding to the training and testing stages. By implementing BN in a continual learning fashion, our approach achieves significant performance gains across a wide range of benchmarks, particularly for the challenging yet realistic online scenarios (e.g., up to 7.68\%, 6.86\% and 4.26\% on Split CIFAR-10, Split CIFAR-100 and Split Mini-ImageNet, respectively).
## Requirement

The code is based on PyTorch framework. To install requirements:

```setup
pip install -r requirements.txt
```
We use [Weights & Biases](https://wandb.ai/site) to log the results. After installing the requirements, type `wandb login` to bind your api key.

## Usage
We provide a launcher to run the experiments conveniently. For instance, use the following command to reproduce the experiments in Table 1 and Table 2 of the paper. 
```shell
python run.py --interpreter `which python` --num_seeds 3
```
During this process, all GPUs will be invoked automatically. It took about 7 hours to complete one seed on our machine (8x NVIDIA RTX A4000).

It is also feasible to run a single experiment manually by `utils/main.py`. The main algorithm is contained in `models/utils/adab2n.py`.


Complete usage of `run.py`:
```
usage: run.py [-h] [--interpreter INTERPRETER] [--coldsecs COLDSECS] [--waitsecs WAITSECS] [--usage_threshold USAGE_THRESHOLD] [--logdir LOGDIR]
              [--models MODELS [MODELS ...]] [--num_seeds NUM_SEEDS] [--bs BS] [--epochs EPOCHS] [--datasets DATASETS [DATASETS ...]] [--methods METHODS [METHODS ...]]
              [--buffer_mode {reservoir,ring}] [--nowandb] [--wandb_project WANDB_PROJECT]

options:
  -h, --help            show this help message and exit
  --interpreter INTERPRETER
                        Interpreter location
  --coldsecs COLDSECS   Seconds to cool down after starting a run
  --waitsecs WAITSECS   Seconds to sleep while waiting for an idle GPU
  --usage_threshold USAGE_THRESHOLD
                        Threshold for determining idle GPUs
  --logdir LOGDIR       The save path for the command line output of each run
  --models MODELS [MODELS ...]
                        Continual learning models to run
  --num_seeds NUM_SEEDS
                        Number of seeds
  --bs BS               Batch size
  --epochs EPOCHS       Number of epochs
  --datasets DATASETS [DATASETS ...]
                        Datasets to run
  --methods METHODS [METHODS ...]
                        Normalization layers to replace BN
  --buffer_mode {reservoir,ring}
                        Replay buffer mode
  --nowandb             Inhibit wandb logging
  --wandb_project WANDB_PROJECT
                        Wandb project name

```

## Acknowledgement

This repository is developed mainly based on the [mammoth](https://github.com/aimagelab/mammoth) repository. Many thanks to its contributors!


## Citation

```bibtex
@article{wang2023hide,
  title={Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation},
  author={Lyu, Yilin and Wang, Liyuan and Zhang, Xingxing and Sun, Zicheng and Su, Hang and Zhu, Jun and Jing, Liping},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```