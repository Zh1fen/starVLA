#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/datasets/ljm_data/zyf/starVLA
LIBERO_HOME=/datasets/ljm_data/zyf/LIBERO
STARVLA_PYTHON=/home/ljm/miniconda3/envs/starVLA/bin/python
LIBERO_PYTHON=/home/ljm/miniconda3/envs/libero/bin/python

# === Checkpoint ===
CKPT=${STARVLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}   
gpu_id=3
port=6694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
