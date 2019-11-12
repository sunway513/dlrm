import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dlrm-dir', type=str, required=True, help="DLRM home directory")
parser.add_argument('--config-size', type=str, required=False, default='small', help='Config sizes', choices=['small', 'medium_4', 'medium_8'])
parser.add_argument('--backend', type=str, required=False, default='pytorch', help="pytorch, caffe2")

args = parser.parse_args()

dlrm_dir = os.path.abspath(args.dlrm_dir)
config_size = args.config_size
backend = args.backend

#if config_size not in ['small', 'medium']:
#    print ("ERROR: Only small and medium are allowed.")
#    sys.exit(1)


sparse_feature_size = '32'
if config_size == 'medium':
    sparse_feature_size = '64'

embedding_size_small = '1000000-' * 31 
embedding_size_small += '1000000'

embedding_size_medium_8 = '10000000-' * 7
embedding_size_medium_8 += '10000000'

embedding_size_medium_4 = '10000000-' * 3
embedding_size_medium_4 += '10000000'

cmd_small = 'python3.6 ' + dlrm_dir + '/' + 'dlrm_s_pytorch.py --arch-sparse-feature-size ' + sparse_feature_size + \
        ' --arch-embedding-size ' + embedding_size_small  + \
        ' --arch-mlp-bot ' + '512-512-512-' + sparse_feature_size + \
        ' --arch-mlp-top ' + '1024-1024-1024-1024-1' + \
        ' --arch-interaction-op ' + 'dot' + \
        ' --activation-function ' + 'relu' + \
        ' --num-batches 1000 ' + \
        ' --num-indices-per-lookup 40 ' + \
        ' --mini-batch-size 200 ' + \
        ' --nepochs 1 ' + \
        ' --use-gpu --print-time 2>&1 | tee log.txt'


cmd_medium_8 = 'python3.6 ' + dlrm_dir + '/' + 'dlrm_s_pytorch.py --arch-sparse-feature-size ' + sparse_feature_size + \
        ' --arch-embedding-size ' + embedding_size_medium_8  + \
        ' --arch-mlp-bot ' + '1600-1024-1024-1024-1024-' + sparse_feature_size + \
        ' --arch-mlp-top ' + '2048-2048-2048-2048-2048-2048-2048-2048-2048-1' + \
        ' --arch-interaction-op ' + 'dot' + \
        ' --activation-function ' + 'relu' + \
        ' --num-batches 1000 ' + \
        ' --num-indices-per-lookup 60 ' + \
        ' --mini-batch-size 512 ' + \
        ' --nepochs 1 ' + \
        ' --use-gpu --print-time 2>&1 | tee log.txt'

cmd_medium_4 = 'python3.6 ' + dlrm_dir + '/' + 'dlrm_s_pytorch.py --arch-sparse-feature-size ' + sparse_feature_size + \
        ' --arch-embedding-size ' + embedding_size_medium_4  + \
        ' --arch-mlp-bot ' + '1600-1024-1024-1024-1024-' + sparse_feature_size + \
        ' --arch-mlp-top ' + '2048-2048-2048-2048-2048-2048-2048-2048-2048-1' + \
        ' --arch-interaction-op ' + 'dot' + \
        ' --activation-function ' + 'relu' + \
        ' --num-batches 1000 ' + \
        ' --num-indices-per-lookup 60 ' + \
        ' --mini-batch-size 512 ' + \
        ' --nepochs 1 ' + \
        ' --use-gpu --print-time 2>&1 | tee log.txt'

caffe2_cmd_small = 'python3.6 ' + dlrm_dir + '/' + 'dlrm_s_caffe2.py --arch-sparse-feature-size ' + sparse_feature_size + \
        ' --arch-embedding-size ' + embedding_size_small  + \
        ' --arch-mlp-bot ' + '512-512-512-' + sparse_feature_size + \
        ' --arch-mlp-top ' + '1024-1024-1024-1024-1' + \
        ' --arch-interaction-op ' + 'dot' + \
        ' --activation-function ' + 'relu' + \
        ' --num-batches 1000 ' + \
        ' --num-indices-per-lookup 40 ' + \
        ' --mini-batch-size 200 ' + \
        ' --nepochs 1 ' + \
        ' --use-gpu --print-time --caffe2-net-type simple 2>&1 | tee log.txt'

caffe2_cmd_medium_8 = 'python3.6 ' + dlrm_dir + '/' + 'dlrm_s_caffe2.py --arch-sparse-feature-size ' + sparse_feature_size + \
        ' --arch-embedding-size ' + embedding_size_medium_8  + \
        ' --arch-mlp-bot ' + '1600-1024-1024-1024-1024-' + sparse_feature_size + \
        ' --arch-mlp-top ' + '2048-2048-2048-2048-2048-2048-2048-2048-2048-1' + \
        ' --arch-interaction-op ' + 'dot' + \
        ' --activation-function ' + 'relu' + \
        ' --num-batches 1000 ' + \
        ' --num-indices-per-lookup 40 ' + \
        ' --mini-batch-size 512 ' + \
        ' --nepochs 1 ' + \
        ' --use-gpu --print-time --caffe2-net-type simple 2>&1 | tee log.txt'

caffe2_cmd_medium_4 = 'python3.6 ' + dlrm_dir + '/' + 'dlrm_s_caffe2.py --arch-sparse-feature-size ' + sparse_feature_size + \
        ' --arch-embedding-size ' + embedding_size_medium_4  + \
        ' --arch-mlp-bot ' + '1600-1024-1024-1024-1024-' + sparse_feature_size + \
        ' --arch-mlp-top ' + '2048-2048-2048-2048-2048-2048-2048-2048-2048-1' + \
        ' --arch-interaction-op ' + 'dot' + \
        ' --activation-function ' + 'relu' + \
        ' --num-batches 1000 ' + \
        ' --num-indices-per-lookup 40 ' + \
        ' --mini-batch-size 512 ' + \
        ' --nepochs 1 ' + \
        ' --use-gpu --print-time --caffe2-net-type simple 2>&1 | tee log.txt'

if config_size == 'small':
    cmd = cmd_small
    if backend == 'caffe2':
        cmd = caffe2_cmd_small
elif config_size == 'medium_4':
    cmd = cmd_medium_4
    if backend == 'caffe2':
        cmd = caffe2_cmd_medium_4
elif config_size == 'medium_8':
    cmd = cmd_medium_8
    if backend == 'caffe2':
        cmd = caffe2_cmd_medium_8

print ("Running ")
print (cmd)
print ("INFO: Running the {} config for dlrm".format(config_size))
os.system(cmd)
print ("OK: Finished Running the {} config".format(config_size))

fs = open('log.txt', 'r')
lines = fs.readlines()
fs.close()

useful_lines = [] 
for j in range(len(lines)):
    if "Finished training" in lines[j]:
        useful_lines.append(lines[j])

total_ms_iter = 0
for j in range(1, len(useful_lines)):
    line = useful_lines[j]
    values = line.split(' ')
    total_ms_iter += float(values[7])

print ("Total ms per iter : {}".format(total_ms_iter))
print ("Avg ms per iter : {}".format(total_ms_iter/(len(useful_lines)- 1)))

