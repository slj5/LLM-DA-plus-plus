#!/bin/bash

set -e

CUDA_DEVICE=0

TEXT_MODELS=(
    clip
)

MOTION_MODELS=(
    dgstgcn
)

LOSSES=(
    info-nce
)

DATASETS=(
    kit-mocap
)

REPS=(
  0
)


# transform into a string of comma-separated values
IFS=,
TEXT_MODELS="${TEXT_MODELS[*]}"
MOTION_MODELS="${MOTION_MODELS[*]}"
DATASETS="${DATASETS[*]}"
LOSSES="${LOSSES[*]}"

# Train & Evaluate
# Perform multiple repetitions of the same experiment, varying the split seed (for cross-validation)

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE HYDRA_FULL_ERROR=1 conda run --no-capture-output -n TBPS python train.py -m ++optim.seed="$REPS" motion_model="$MOTION_MODELS" text_model="$TEXT_MODELS" optim="$LOSSES" data="$DATASETS" || true
# HYDRA_FULL_ERROR=1 python evaluate.py runs/experiment=$EXP/run-$REP --debug 

