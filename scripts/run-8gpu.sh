#!/bin/bash
pip3 install onnx
cp *.sh ../ && cd ..
bash pytorch_dlrm_py3.6_1_gpu_medium_4.sh |& tee   1_gpu_medium_4 
sleep 5
bash pytorch_dlrm_py3.6_1_gpu_medium_8.sh |& tee   1_gpu_medium_8
sleep 5
bash pytorch_dlrm_py3.6_1_gpu_small_4.sh |& tee    1_gpu_small_4
sleep 5
bash pytorch_dlrm_py3.6_1_gpu_small_8.sh |& tee    1_gpu_small_8
sleep 5
bash pytorch_dlrm_py3.6_2_gpu_medium_16.sh |& tee  2_gpu_medium_16
sleep 5
bash pytorch_dlrm_py3.6_2_gpu_medium_8.sh |& tee   2_gpu_medium_8
sleep 5
bash pytorch_dlrm_py3.6_2_gpu_small_8.sh |& tee    2_gpu_small_8
sleep 5
bash pytorch_dlrm_py3.6_2_gpu_small_16.sh |& tee    2_gpu_small_16
sleep 5
bash pytorch_dlrm_py3.6_4_gpu_medium_16.sh |& tee  4_gpu_medium_16
sleep 5
bash pytorch_dlrm_py3.6_4_gpu_medium_32.sh |& tee  4_gpu_medium_32
sleep 5
bash pytorch_dlrm_py3.6_4_gpu_small_16.sh |& tee   4_gpu_small_16
sleep 5
bash pytorch_dlrm_py3.6_4_gpu_small_32.sh |& tee   4_gpu_small_32
sleep 5
bash pytorch_dlrm_py3.6_8_gpu_medium_32.sh |& tee  8_gpu_medium_32
sleep 5
bash pytorch_dlrm_py3.6_8_gpu_medium_64.sh |& tee  8_gpu_medium_64
sleep 5
bash pytorch_dlrm_py3.6_8_gpu_small_32.sh |& tee   8_gpu_small_32
sleep 5
bash pytorch_dlrm_py3.6_8_gpu_small_64.sh |& tee   8_gpu_small_64
sleep 5
rm *.sh
