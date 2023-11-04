#!/bin/bash

dir_path="../log/ML100K/"

if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
fi

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=1234 ../src/train.py --item_indexing sequential --tasks sequential,straightforward --datasets ML100K --epochs 10 --batch_size 128 --backbone t5-small --cutoff 1024 > ../log/ML100K/ML100K_t5_sequential.log
