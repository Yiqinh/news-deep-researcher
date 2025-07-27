#!/bin/sh

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi

source conda activate CIR
cd project/jonmay_231/spangher/news-deep-researcher

pip install -qU langchain-community faiss-gpu
python scripts/embed_sources.py