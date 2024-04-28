python help.py \
    --image_dir /home/liyaw22/R2GenCMN/data/mimic_cxr/images/ \
    --ann_path /home/liyaw22/R2GenCMN/data/mimic_cxr/annotation.json \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 10 \
    --batch_size 16 \
    --epochs 30 \
    --step_size 1 \
    --gamma 0.8