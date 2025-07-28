#!/bin/sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=80GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi

conda activate deepresearch
cd /project/jonmay_231/spangher/news-deep-researcher

python scripts/embed_sources.py