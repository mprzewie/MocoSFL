#!/bin/bash
#SBATCH --job-name=fed_c200_9f_niid_s1
#SBATCH --gres=gpu:1
##SBATCH --mem-per-gpu=40G
#SBATCH --mem=90G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=45
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

#cd "$(dirname "$0")"
#cd ../../

#fixed arguments
num_epoch=200
lr=0.03
moco_version=V2
arch=MobileNetV2

seed_list="1"
non_iid_list="0.2"
cutlayer_list="9"

break_epoch=-1
num_client_list="200"
dataset_list="cifar10"
loss_threshold=0.0
ressfl_alpha=0.0
bottleneck_option=None

#python prepare_imagenet12.py

for dataset in $dataset_list; do
        for seed in $seed_list; do
                for num_client in $num_client_list; do
                        for noniid_ratio in $non_iid_list; do
                                for cutlayer in $cutlayer_list; do
                                        output_dir="./outputs/imagenet12_cut2/mocosfl${moco_version}_${arch}_${dataset}_cut${cutlayer}_bnl${bottleneck_option}_client${num_client}_nonIID${noniid_ratio}_seed_${seed}"
                                        python run_sflmoco.py --num_client ${num_client} --lr ${lr} --cutlayer ${cutlayer} --num_epoch ${num_epoch}\
                                                --noniid_ratio ${noniid_ratio} --output_dir ${output_dir}\
                                                --moco_version ${moco_version} --arch ${arch} --dataset ${dataset} --loss_threshold ${loss_threshold}\
                                                --ressfl_alpha ${ressfl_alpha} --bottleneck_option ${bottleneck_option} --auto_adjust --seed ${seed} # --resume #--attack #--load_model ${output_dir} --break_epoch ${break_epoch}
                                done
                        done
                done
        done
done
