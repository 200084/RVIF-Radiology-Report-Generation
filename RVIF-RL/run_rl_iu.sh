seed=${RANDOM}
#seed=16573
seed0=13781

mkdir -p results/iu_xray-Vencode/base_cmn_rl/
mkdir -p records/iu_xray-Vencode/base_cmn_rl/

CUDA_VISIBLE_DEVICES=2 python train_rl.py \
--image_dir /home/liyaw22/R2GenCMN/data/iu_xray/images/ \
--ann_path /home/liyaw22/R2GenCMN/data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--batch_size 6 \
--epochs 200 \
--save_dir results/iu_xray-Vencode/base_cmn_rl-ablation/ \
--record_dir records/iu_xray-Vencode/base_cmn_rl2-ablation/ \
--step_size 1 \
--gamma 0.8 \
--seed ${seed} \
--topk 32 \
--beam_size 3 \
--log_period 100 \
--resume results/iu_xray-Vencode/_seed_13781/current_checkpoint40-best.pth

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
