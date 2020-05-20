#!/bin/bash

# the following variable embedding_size evaluates to 10000000-10000000-10000000....64 times.
# where 10000000 is the average number of rows per table and 64 is the number of embedding tables for medium_64 config
embedding_size=$(printf '10000000-%.0s' {1..63})
embedding_size+=10000000

sparse_feature_size=64

HIP_VISIBLE_DEVICES=0,1,2,3 python3.6 dlrm_s_pytorch.py \
--arch-sparse-feature-size $sparse_feature_size \
--arch-embedding-size $embedding_size \
--arch-mlp-bot "1600-1024-1024-1024-1024-$sparse_feature_size" \
--arch-mlp-top "2048-2048-2048-2048-2048-2048-2048-2048-2048-1" \
--arch-interaction-op "dot" \
--activation-function "relu" \
--num-batches 200 \
--num-indices-per-lookup 60 \
--mini-batch-size 4096 \
--nepochs 1 \
--use-gpu \
--print-time \
--emulate-8-gpu \
