#!/bin/bash
#SBATCH --job-name=FedSSL
#SBATCH --gpus=1g.10gb:1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1

set -e

eval "$(conda shell.bash hook)"
conda activate uj

set -x

#cd "$(dirname "$0")"
#cd ../../

#export WANDB__SERVICE_WAIT=300
#export WANDB_API_KEY=8922102d08435f66d8640bbfa9caefd9c4e6be6d
#export WANDB_PROJECT=federated_ssl
#export WANDB_ENTITY=gmum

#fixed arguments
num_epoch=200
lr=0.06
moco_version=V2
arch=ResNet18

#non_iid_list="0.2 1.0"
noniid_ratio="0.2" # 1.0"
cutlayer_list="1" # 2"
num_client=5
dataset=cifar100
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None

prefix="mocosfl${moco_version}_${arch}_${dataset}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}"
constant_args="--num_client ${num_client} --lr ${lr} --num_epoch ${num_epoch} --noniid_ratio ${noniid_ratio}  --moco_version ${moco_version} \
  --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold} --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option}
  --auto_adjust --divergence_measure"

#cutlayer=1

for cutlayer in 1;
do

#output_dir="./outputs/${prefix}_cut${cutlayer}_baseline"
#python run_sflmoco.py  $constant_args --cutlayer ${cutlayer} --output_dir ${output_dir}
#
#output_dir="./outputs/${prefix}_cut${cutlayer}_no-ft-sharing"
#python run_sflmoco.py  $constant_args --cutlayer ${cutlayer} --output_dir ${output_dir} --disable_feature_sharing
#
#output_dir="./outputs/${prefix}_cut${cutlayer}_div-aware"
#python run_sflmoco.py  $constant_args --cutlayer ${cutlayer} --output_dir ${output_dir} --divergence_aware
#
output_dir="./outputs/${prefix}_cut${cutlayer}_div-aware_no-ft-sharing"
python run_sflmoco.py  $constant_args --cutlayer ${cutlayer} --output_dir ${output_dir} --disable_feature_sharing --divergence_aware

done