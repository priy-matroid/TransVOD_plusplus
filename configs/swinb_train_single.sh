#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=./experiments/CBP_single_id_100/
mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 30 \
    --lr_drop_epochs 21 28 \
    --num_feature_levels 1\
    --num_queries 100 \
    --dilation \
    --batch_size 4 \
    --hidden_dim 256 \
    --num_workers 24 \
    --with_box_refine \
    --img_side 600 \
    --num_classes 9 \
    --resume ./pretrained_checkpoints/single/transvod_pp_single_swinb_checkpoint0006.pth \
    --coco_pretrain \
    --data_root /home/ubuntu/priy_dev/Datasets/VisDrone/ \
    --dataset_file 'vid_single' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
