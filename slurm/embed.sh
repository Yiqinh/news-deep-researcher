#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

python scripts/embed_sources.py