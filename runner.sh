#!/bin/bash

# Convert 'T' to 'True' and 'F' to 'False'
if [ "$1" = "T" ]; then
    gpu_enabled="True"
elif [ "$1" = "F" ]; then
    gpu_enabled="False"
else
    echo "Invalid GPU argument. Use 'T' for True or 'F' for False."
    exit 1
fi

# Extract other arguments
hour=$2               # Number of hours
cpu_number=$3         # Number of CPUs
gpu_number=$4         # Number of GPUs (only used if gpu_enabled is "True")

# Base srun command
base_command="srun --time=${hour}:00:00 --nodes=1 --cpus-per-task=${cpu_number} --pty /usr/bin/bash"

# Check if GPU is enabled
if [ "$gpu_enabled" = "True" ]; then
    # Run with GPU settings
    eval "$base_command --partition=gpu-interactive --gres=gpu:$gpu_number"
else
    # Run without GPU settings
    eval "$base_command"
fi

module load miniconda3/22.11.1-hydt3qz
source activate mahdi
exec zsh
source ~/.zshrc
