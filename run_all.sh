#!/bin/bash

# Check if environment name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <environment_name>"
  exit 1
fi

ENV_NAME=$1

# Activate conda environment
# Try to find conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Activate the environment
# We use 'conda activate' if available, otherwise assume it's already active or we hope for the best
if command -v conda >/dev/null 2>&1; then
    conda activate curiosity-pytorch
else
    echo "Conda not found or not initialized. Assuming 'curiosity-pytorch' environment is already active or python is correct."
fi

echo "Starting runs for environment: $ENV_NAME"

# Run 4 algorithms on 4 different GPUs
# We use the --device argument we added to run.py

echo "Launching Inverse Dynamics on cuda:0"
python run.py --config configs/inverse_dynamics.yaml --env "$ENV_NAME" --device cuda:0 --name "inverse_dynamics_${ENV_NAME}" &

echo "Launching Pixels on cuda:1"
python run.py --config configs/pixels.yaml --env "$ENV_NAME" --device cuda:1 --name "pixels_${ENV_NAME}" &

echo "Launching Random Features on cuda:2"
python run.py --config configs/random_features.yaml --env "$ENV_NAME" --device cuda:2 --name "random_features_${ENV_NAME}" &

echo "Launching VAE on cuda:3"
python run.py --config configs/vae.yaml --env "$ENV_NAME" --device cuda:3 --name "vae_${ENV_NAME}" &

# Wait for all background processes to finish
wait

echo "All runs completed."
