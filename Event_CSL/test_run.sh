#!/bin/bash  

export PYTHONWARNINGS=ignore::UserWarning
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=2125 --use_env train_slt.py \
--batch-size 8 \
--num_workers 8 \
--epochs 200 \
--opt sgd \
--lr 0.01 \
--output_dir result_test/event-csl \
--eval 

