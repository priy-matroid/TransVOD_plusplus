#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=./experiments/Visdrone_overfit_multi/
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 100 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --hidden_dim 256 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume /home/ubuntu/priy_dev/TransVOD/TransVOD_plusplus/experiments/VisDrone_vid_single_overfit/checkpoint0010.pth \
    --lr_drop_epochs 75 \
    --num_workers 24 \
    --with_box_refine \
    --img_side 600 \
    --num_classes 10 \
    --dataset_file 'vid_multi' \
    --data_root /home/ubuntu/priy_dev/Datasets/VisDrone \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T

# --vid_path
# 