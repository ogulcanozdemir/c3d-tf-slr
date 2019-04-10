#!/bin/bash
#
#SBATCH --partition=single
#SBATCH --account ogozdemir
#SBATCH --job-name compute-vol-mean
#SBATCH --nodes 1          # nodes requested
#SBATCH --ntasks-per-node 1          # tasks requested
#SBATCH --cpus-per-task 1          # cores requested
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu 5000
#SBATCH --time=01-00:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH --output /truba/home/ogozdemir/code/c3d-tf-slr/results/out-%j.out  # send stdout to outfile
#SBATCH --error /truba/home/ogozdemir/code/c3d-tf-slr/results/err-%j.err  # send stderr to errfile

/truba/home/ogozdemir/anaconda3/envs/tf2-gpu/bin/python compute_volume_mean.py
