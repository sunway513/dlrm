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
rm *.sh
