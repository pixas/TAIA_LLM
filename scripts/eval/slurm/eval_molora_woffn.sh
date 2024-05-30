#!/bin/bash

# log path and experiments name
LOGS_BASE_PATH="./logs/"
CKPT=qwen1.5-1.8b-lora-woffn

# base model path, lora weight and conversation formate
MODEL_BASE=<Qwen1.5-1.8b-chat-path>
MODEL_PATH=<checkpoint_path>
conv_mode="qwen"

# evaluated dataset
domain="math"

echo "Processing $domain"
mkdir -p ${LOGS_BASE_PATH}/${domain}
srun -p partition --gres=gpu:1  --quotatype=auto --output=${LOGS_BASE_PATH}/${domain}/${CKPT}.infer.log python -m ming.eval.model_diverse_gen \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --question-file datas/eval/${domain}.json \
    --answers-file ${LOGS_BASE_PATH}/${domain}/${CKPT}.jsonl \
    --temperature 0 \
    --max-tokens 1024 \
    --conv-mode ${conv_mode} \
    --infer-answer \
    --use-logit-bias \
    --infer-answer \
    --resume

echo "Evaluating $domain"
srun -p partition --output=${LOGS_BASE_PATH}/${domain}/${CKPT}.eval.log python -m ming.eval.eval_em \
    --input_file ${LOGS_BASE_PATH}/${domain}/${CKPT}.jsonl 

wait
echo "All processes are done."