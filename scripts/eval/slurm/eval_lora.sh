#!/bin/bash

# log path and experiments name
LOGS_BASE_PATH="./logs/"


# base model path, lora weight and conversation formate
MODEL_BASE=<Qwen1.5-1.8b-chat-path> # replace this as your actual model path; we support using the Qwen 1.5 family
MODEL_PATH=<checkpoint_path> # replace the checkpoint as your actual checkpoint folder
conv_mode="qwen"

# evaluated dataset
domain="math" # we take math as an example, but you can replace it to other domains
# we support evaluation on the `datas/eval` folder, which contains the evaluation data for each domain

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