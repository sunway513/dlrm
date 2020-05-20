#!/bin/bash

# the following variable embedding_size evaluates to 1000000-1000000-1000000....64 times.
# where 1000000 is the average number of rows per table and 64 is the number of embedding tables for small config
embedding_size=$(printf '1000000-%.0s' {1..63})
embedding_size+=1000000

sparse_feature_size=32

HIP_VISIBLE_DEVICES=0 python3.6 dlrm_s_pytorch.py \
--arch-sparse-feature-size $sparse_feature_size \
--arch-embedding-size $embedding_size \
--arch-mlp-bot "512-512-512-$sparse_feature_size" \
--arch-mlp-top "1024-1024-1024-1024-1" \
--arch-interaction-op "dot" \
--activation-function "relu" \
--num-batches 200 \
--num-indices-per-lookup 40 \
--mini-batch-size 1600 \
--nepochs 1 \
--use-gpu \
--print-time \
