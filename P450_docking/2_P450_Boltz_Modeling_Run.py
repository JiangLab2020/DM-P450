import os
import sys
import re

#######################################################################################################   selecting the GPU
GPU_Num = -1
All_GPUs = 1
os.system('nvidia-smi >nvidia.txt')
nvidia_dict = {}
with open('nvidia.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if re.match(r'\|   (\d+?) .*GeForce',line):
            All_GPUs = re.match(r'\|   (\d+?) .*GeForce', line).group(1)
        if re.match(r'\|    (\d)   N\/A  N\/A',line):
            num = re.match(r'\|    (\d)   N\/A  N\/A',line).group(1)
            nvidia_dict[num] = 1
for i in range(int(All_GPUs)+1):
    if str(i) not in nvidia_dict:
        GPU_Num = i
        break
print('there are totally ' + str(int(All_GPUs)+1) + ' GPUs')
if i > -1:
    print('selecting the GPU:' + str(GPU_Num))
    os.environ['CUDA_VISIBLE_DEVICES']=str(GPU_Num)
else:
    print('on free GPU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#######################################################################################################   selecting the GPU


json_dir = sys.argv[1]

gpu_num = 1
if len(sys.argv) > 2:
    gpu_num = sys.argv[2]

out_dir = json_dir + '-Out'

os.system(f'boltz predict {json_dir} --output_format pdb --out_dir {out_dir} --devices {gpu_num}')
