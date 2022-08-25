#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

REPO=$PWD

LAN=${1:-"en"}
ENCODER_MODEL=${2:-"xlm-roberta-base"}
DATA_DIR=${3:-"$REPO/data/"}
OUT_DIR=${4:-"$REPO/output/"}

model_name="ner_${LAN}"
base_dir=${REPO}/../
train_file=${DATA_DIR}/train.txt
dev_file=${DATA_DIR}/dev.txt
gazetteer_path=${base_dir}/gazetteer_demo/${LAN}

ckpt_file_path=${OUT_DIR}/ner_${LAN}

python -m fine_tune --train "$train_file" --dev "$dev_file" --test "$test_file" --gazetteer "$gazetteer_path" \
                           --out_dir "$OUT_DIR" --model_name "$model_name" --gpus 1 --epochs 20 --encoder_model "$ENCODER_MODEL" \
                           --model "$ckpt_file_path"  --batch_size 32 --lr 0.00002

