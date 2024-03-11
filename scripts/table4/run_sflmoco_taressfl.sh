#!/bin/bash
#cd "$(dirname "$0")"
#cd ../../

#SBATCH --job-name=fed_attack_cut_9_02.0_01
#SBATCH --gres=gpu:1
##SBATCH --mem-per-gpu=40G
#SBATCH --mem=90G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1 
#SBATCH -A plgplgccontrastive-gpu-a100
#SBATCH --time=48:00:00

set -e
set -x

cd $SCRATCH/MocoSFL/

export WANDB_API_KEY=f61fe6de67dc18515ebe11ca944faaa2ccdd11e1
export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=fed_bug
export WANDB_ENTITY=gmum


#fixed arguments
num_epoch=200
lr=0.001
c_lr=0.00
moco_version=V2
arch=ResNet18
seed=1234
aux_data=cifar100
non_iid_list="0.2" #1.0
cutlayer_list="9"
num_client=100
dataset=cifar10
loss_threshold=0.0
ressfl_alpha=2.0
bottleneck_option_list="C4S2"
data_proportion_list="0.01" # 0.01 0.005"
ressfl_target_ssim_list="0.6"
for cutlayer in $cutlayer_list; do
        for ressfl_target_ssim in $ressfl_target_ssim_list; do
                for bottleneck_option in $bottleneck_option_list; do
                        for data_proportion in $data_proportion_list; do
                                for noniid_ratio in $non_iid_list; do
                                        initialze_path="./expert_target_aware/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_expert_batchsize128_ressfl_2.0_SSIM${ressfl_target_ssim}_data_proportion_${data_proportion}_aux_${aux_data}"
                                        output_dir="./outputs/ressfl_freeze_target_aware/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}_initialize_CLR${c_lr}_SSIM${ressfl_target_ssim}_data_proportion_${data_proportion}"
                                        CUDA_VISIBLE_DEVICES=0 python run_sflmoco.py --num_client ${num_client} --lr ${lr} --c_lr ${c_lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold} --load_server\
                                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --auto_adjust --seed ${seed} --initialze_path ${initialze_path} --resume --attack
                                done
                        done
                done
        done
done
## for test, add --resume --attack
