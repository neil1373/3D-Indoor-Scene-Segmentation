#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export EXPERIMENT=$2
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-Scannet200Voxelization2cmDataset}
export MODEL=${MODEL:-Res16UNet34A}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-1}
export BATCH_SIZE=${BATCH_SIZE:-2}
export SCHEDULER=${SCHEDULER:-SquaredLR}
export MAX_ITER=${MAX_ITER:-60000}
export ITER_SIZE=${ITER_SIZE:-6}

export OUTPATH=./outputs/$DATASET/$MODEL/${OPTIMIZER}-l$LR-b$BATCH_SIZE-$SCHEDULER-i$MAX_ITER-$EXPERIMENT/$TIME
export VERSION=$(git rev-parse HEAD)

# Save the experiment detail and dir to the common log file
mkdir -p $OUTPATH

LOG="$OUTPATH/$TIME.txt"

# put the arguments on the first line for easy resume
echo -e "
    --log_dir $OUTPATH \
    --dataset $DATASET \
    --model $MODEL \
    --train_limit_numpoints 1200000 \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --iter_size $ITER_SIZE \
    --scheduler $SCHEDULER \
    --max_iter $MAX_ITER \
    $3" >> $LOG
echo Logging output to "$LOG"
echo $(pwd) >> $LOG
echo "Version: " $VERSION >> $LOG
echo "Git diff" >> $LOG
echo "" >> $LOG
git diff | tee -a $LOG
echo "" >> $LOG
nvidia-smi | tee -a $LOG

time python -W ignore spatio_temporal/main.py \
    --log_dir $OUTPATH \
    --dataset $DATASET \
    --model $MODEL \
    --train_limit_numpoints 1200000 \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --iter_size $ITER_SIZE \
    --scheduler $SCHEDULER \
    --max_iter $MAX_ITER \
    $3 2>&1 | tee -a "$LOG"