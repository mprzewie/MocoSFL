#!/bin/bash

export WANDB_API_KEY=
export WANDB__SERVICE_WAIT=
export WANDB_PROJECT=
export WANDB_ENTITY=

#fixed arguments
num_epoch=200
lr=0.06
moco_version=V2
arch=MobileNetV2


non_iid_list="0.2"
cutlayer_list="1 2 3 4 5 6 7 8 9"
num_client_list="20"
dataset=cifar10 # or cifar100
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None
batch_size=20
client_sample_ratio=0.25
avg_freq=10


for num_client in $num_client_list; do
        for noniid_ratio in $non_iid_list; do
                for cutlayer in $cutlayer_list; do
                        output_dir="./outputs/monacosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}_af${avg_freq}"
                        python run_sflmoco_repro.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --batch_size ${batch_size}\
                                --client_sample_ratio ${client_sample_ratio} --avg_freq ${avg_freq} \
                                --divergence_measure --seed 1138 --symmetric_original --fedavg-momentum
                done
        done
done