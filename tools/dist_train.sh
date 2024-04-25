#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch > log.txt&
