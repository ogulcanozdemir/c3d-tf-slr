#!/bin/bash
#
#SBATCH --partition=akya-cuda
#SBATCH --account ogozdemir
#SBATCH --job-name tf-gpu-test
#SBATCH --nodes 1         # nodes requested
#SBATCH --ntasks-per-node 1          # tasks requested
#SBATCH --cpus-per-task 20          # cores requested
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu 15000
#SBATCH --time=02-00:00:00
#SBATCH --mail-type ALL
#SBATCH --mail-user ogulcan.ozdemir@yahoo.com
#SBATCH --output /truba/home/ogozdemir/code/c3d-tf-slr/scripts/results/outputs/out-%j.out  # send stdout to outfile
#SBATCH --error /truba/home/ogozdemir/code/c3d-tf-slr/scripts/results/errors/err-%j.err  # send stderr to errfile

module load centos7.3/lib/cuda/10.0

/truba/home/ogozdemir/anaconda3/envs/tf2-gpu/bin/python train_c3d.py \
							--max-steps=1000000 \
							--batch-size=30 \
							--num-classes=174 \
							--crop-size=112 \
							--channels=3 \
							--num-frames-per-clip=16 \
							--model-save-dir=checkpoints/c3d_general_model \
							--moving-average-decay=0.9999 \
							--num-gpu=2 \
							--train-list=list/general/train.list \
							--test-list=list/general/test.list \
							--crop-mean=models/crop_mean_16_generalv2.npy \
							--pretrained-weights=models/conv3d_deepnetA_sport1m_iter_1900000_TF.model
