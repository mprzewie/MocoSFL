#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate uj

set -x

cd "$(dirname "$0")"
cd ../../

#export WANDB_API_KEY=8922102d08435f66d8640bbfa9caefd9c4e6be6d
#export WANDB_PROJECT=federated_ssl
#export WANDB_ENTITY=gmum

#fixed arguments
num_epoch=200
lr=0.06
moco_version=V2
arch=ResNet18

#non_iid_list="0.2 1.0"
batch_size=128
non_iid_list="0.2" # 1.0"
cutlayer_list="1" # 2"
num_client_list="20"
dataset=cifar100
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None
for num_client in $num_client_list; do
        for noniid_ratio in $non_iid_list; do
                for cutlayer in $cutlayer_list; do
                        output_dir="./outputs/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}"
                        python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option}  --auto_adjust  --disable_feature_sharing
                done
        done
done

#non_iid_list="0.2 1.0"
#cutlayer_list="1 2"
#num_client_list="20"
#dataset=cifar100
#loss_threshold=0.0
#ressfl_alpha=0.0
#bottleneck_option=None
#batch_size=20
#client_sample_ratio=0.25
#avg_freq=10
#for num_client in $num_client_list; do
#        for noniid_ratio in $non_iid_list; do
#                for cutlayer in $cutlayer_list; do
#                        output_dir="./outputs/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}"
#                        python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
#                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
#                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
#                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --batch_size ${batch_size}\
#                                --client_sample_ratio ${client_sample_ratio} --avg_freq ${avg_freq}
#                done
#        done
#done
## for test, add --resume --attack