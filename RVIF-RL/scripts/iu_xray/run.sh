#!/bin/bash
#SBATCH -J base+rl
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

#source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh

#conda activate a100

#seed=23838
seed=${RANDOM}
noamopt_warmup=1000

CUDA_VISIBLE_DEVICES=0 python train.py\
    --image_dir /home/liyaw22/R2GenCMN/data/iu_xray/images/ \
    --ann_path /home/liyaw22/R2GenCMN/data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 120 \
    --batch_size 16 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed ${seed} \
    --beam_size 3 \
    --save_dir results/iu_xray-Vencode/ \
    --log_period 80

#RESUME=results/iu_xray/base_seed_${seed}

#seed=${RANDOM}
#noamopt_warmup=1000
#save_dir=results/iu_xray/rl_base_seed_${seed}
#echo "seed ${seed}"

#python train_rl_base.py \
#    --image_dir /home/liyaw22/R2GenCMN/data/iu_xray/images/ \
#    --ann_path /home/liyaw22/R2GenCMN/data/iu_xray/annotation.json \
#    --dataset_name iu_xray \
#    --max_seq_length 100 \
#    --threshold 10 \
#    --batch_size 6 \
#    --epochs 50 \
#    --save_dir ${save_dir} \
#    --step_size 1 \
#    --gamma 0.8 \
#    --seed ${seed} \
#    --topk 32 \
#    --sc_eval_period 3000 \
#    --resume ${RESUME}/current_checkpoint.pth
