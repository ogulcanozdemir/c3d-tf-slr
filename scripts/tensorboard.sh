#!/bin/bash
#
#SBATCH --partition=akya-cuda
#SBATCH --account ogozdemir
#SBATCH --job-name compute-vol-mean
#SBATCH --nodes 1          # nodes requested
#SBATCH --ntasks-per-node 1          # tasks requested
#SBATCH --cpus-per-task 20          # cores requested
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu 10000
#SBATCH --time=01-00:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH --output /truba/home/ogozdemir/code/c3d-tf-slr/results/out-%j.out  # send stdout to outfile
#SBATCH --error /truba/home/ogozdemir/code/c3d-tf-slr/results/err-%j.err  # send stderr to errfile

/truba/home/ogozdemir/anaconda3/envs/tf2-gpu/bin/tensorboard --logdir=/truba/home/ogozdemir/code/c3d-tf-slr/checkpoints/c3d_general_model/visual_logs

