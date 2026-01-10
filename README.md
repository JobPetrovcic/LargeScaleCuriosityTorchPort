# Large Scale Curiosity Torch Port

A PyTorch implementation of the [Large Scale Curiosity](https://arxiv.org/pdf/1808.04355) paper ([Large Scale Curiosity](https://github.com/openai/large-scale-curiosity) implementation in TensorFlow), implementing curiosity-driven exploration with auxiliary tasks for reinforcement learning. This repository attempts to replicate the exact functional behavior of the TensorFlow implementation, including architecture, algorithm, weight initialization, normalization, and entropy logarithm base. We made changes to logging (using wandb) and multiprocessing. The code relies on `AsyncVectorEnv` rather than the custom multiprocessing logic found in the original repository. This should not functionally affect the resulting model and training. 

However, to exactly replicate the experiments, one would also have to replicate the seed and randomized tensor operations from TensorFlow. It is unclear if this is even possible, but given that the results are similar in terms of performance, there is little benefit in doing so.

Note that I have not tested all environments yet.

### Functional differences

If you find a functional difference, which could potentially affect performance, please open an issue or a PR. 

## Installation and Dependencies

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate curiosity-pytorch
```

The dependencies are listed in `environment.yml`. The exact environment state `frozen_environment_snapshot.yml` is also provided for reference.

## Usage

Run a single experiment:

```bash
python run.py --config configs/config.yaml --env <environment_name>
```

Run all auxiliary tasks on different GPUs:

```bash
./run_all.sh <environment_name>
```

Available configurations:
- `configs/inverse_dynamics.yaml`
- `configs/pixels.yaml`
- `configs/random_features.yaml`
- `configs/vae.yaml`
