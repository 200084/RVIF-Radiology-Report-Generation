#!/bin/bash
#SBATCH -J base+rl
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

#source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh

#conda activate a100

seed=23838
seed0=9153
# seed=${RANDOM}
noamopt_warmup=1000

# python train_base.py \
#     --image_dir data/mimic_cxr/images/ \
#     --ann_path data/mimic_cxr/annotation.json \
#     --dataset_name mimic_cxr \
#     --max_seq_length 100 \
#     --threshold 10 \
#     --batch_size 16 \
#     --epochs 30 \
#     --save_dir results/mimic_cxr/base_seed_${seed} \
#     --step_size 1 \
#     --gamma 0.8 \
#     --seed ${seed}

RESUME=results/mimic_cxr-Vencode/base_seed_${seed0}

seed=${RANDOM}
noamopt_warmup=1000
save_dir=results/mimic_cxr-Vencode/rl_base_seed_${seed}
echo "seed ${seed}"

CUDA_VISIBLE_DEVICES=3 python train_rl_base.py \
    --image_dir /home/liyaw22/R2GenCMN/data/mimic_cxr/images/ \
    --ann_path /home/liyaw22/R2GenCMN/data/mimic_cxr/annotation.json \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 10 \
    --batch_size 16 \
    --epochs 50 \
    --save_dir ${save_dir} \
    --step_size 1 \
    --gamma 0.8 \
    --seed ${seed} \
    --topk 32 \
    --sc_eval_period 3000 \
    --resume ${RESUME}/model_best.pth
#    --resume ${RESUME}/current_checkpoint.pth
