#!/bin/bash
#SBATCH --job-name=fed_c20_9_niid_mobilenet_re
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1 
#SBATCH -A plgplgccontrastive-gpu-a100
#SBATCH --time=48:00:00

set -e
set -x

cd $SCRATCH/MocoSFL/
source /net/tscratch/people/plgmarcinosial/miniconda3/bin/activate /net/tscratch/people/plgmarcinosial/miniconda3/envs/mae_env

#cd "$(dirname "$0")"
#cd ../../


export WANDB_API_KEY=f61fe6de67dc18515ebe11ca944faaa2ccdd11e1
export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=fed_bug
export WANDB_ENTITY=gmum

#fixed arguments
num_epoch=200
lr=0.06
moco_version=V2
arch=MobileNetV2
seed=1

#non_iid_list="0.2"
#cutlayer_list="1"
#num_client_list="5"
#dataset=cifar10
#loss_threshold=0.0
#ressfl_alpha=0.0
#bottleneck_option=None
#for num_client in $num_client_list; do
#        for noniid_ratio in $non_iid_list; do
#                for cutlayer in $cutlayer_list; do
#                        output_dir="./outputs/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}"
#                        python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
#                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
#                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
#                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --auto_adjust --seed ${seed} --resume --attack
#                done
#        done
#done


non_iid_list="0.2"
cutlayer_list="9"
num_client_list="20"
dataset=cifar10
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None
batch_size=20
client_sample_ratio=0.25
avg_freq=10
for num_client in $num_client_list; do
        for noniid_ratio in $non_iid_list; do
                for cutlayer in $cutlayer_list; do
                        output_dir="./outputs/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}"
                        python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --batch_size ${batch_size}\
                                --client_sample_ratio ${client_sample_ratio} --avg_freq ${avg_freq} --seed ${seed} --resume # --attack
                done
        done
done

# for test, add --resume --attack
