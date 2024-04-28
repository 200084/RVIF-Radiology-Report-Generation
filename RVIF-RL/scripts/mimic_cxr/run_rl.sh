seed=${RANDOM}
seed0=9153

mkdir -p results/mimic_cxr/base_cmn_rl/
mkdir -p records/mimic_cxr/base_cmn_rl/

CUDA_VISIBLE_DEVICES=3 python train_rl.py \
--image_dir /home/liyaw22/R2GenCMN/data/mimic_cxr/images/ \
--ann_path /home/liyaw22/R2GenCMN/data/mimic_cxr/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 6 \
--epochs 50 \
--save_dir results/mimic_cxr/base_cmn_rl/ \
--record_dir records/mimic_cxr/base_cmn_rl/ \
--step_size 1 \
--gamma 0.8 \
--seed ${seed} \
--topk 32 \
--sc_eval_period 3000 \
--resume results/mimic_cxr-Vencode/base_seed_${seed0}/model_best.pth

# python train_rl.py \
# --image_dir data/mimic_cxr/images/ \
# --ann_path data/mimic_cxr/annotation.json \
# --dataset_name mimic_cxr \
# --max_seq_length 100 \
# --threshold 10 \
# --batch_size 6 \
# --epochs 50 \
# --save_dir results/mimic_cxr/base_cmn_rl/ \
# --record_dir records/mimic_cxr/base_cmn_rl/ \
# --step_size 1 \
# --gamma 0.8 \
# --seed ${seed} \
# --topk 32 \
# --sc_eval_period 3000 \
# --resume results/mimic_cxr/mimic_cxr_0.8_1_16_5e-5_1e-4_3_3_32_2048_512_30799/current_checkpoint.pth
