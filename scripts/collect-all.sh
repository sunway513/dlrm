#!/bin/bash
echo "DLRM performance measured in number of iters/sec"
cd ..
echo "1 GPU numbers"
cat 1_gpu_small_4 | grep "Average num"
cat 1_gpu_small_8 | grep "Average num"
cat 1_gpu_medium_4 | grep "Average num"
cat 1_gpu_medium_8 | grep "Average num"

echo "2 GPU numbers"
cat 2_gpu_small_8 | grep "Average num"
cat 2_gpu_small_16 | grep "Average num"
cat 2_gpu_medium_8 | grep "Average num"
cat 2_gpu_medium_16 | grep "Average num"

echo "4 GPU numbers"
cat 4_gpu_small_16 | grep "Average num"
cat 4_gpu_small_32 | grep "Average num"
cat 4_gpu_medium_16 | grep "Average num"
cat 4_gpu_medium_32 | grep "Average num"

echo "8 GPU numbers"
cat 8_gpu_small_32 | grep "Average num"
cat 8_gpu_small_64 | grep "Average num"
cat 8_gpu_medium_32 | grep "Average num"
cat 8_gpu_medium_64 | grep "Average num"
