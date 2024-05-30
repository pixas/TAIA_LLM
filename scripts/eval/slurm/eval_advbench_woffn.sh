#!/bin/bash


domain=advbench
LOGS_BASE_PATH="./logs/diverse"
MODEL_BASE=<llama2_chat_path>
MODEL_PATH=<checkpoint_path>
if [[ $MODEL_PATH == *"llama7b"* ]]; then

    if [[ $CKMODEL_PATHPT == *"aoa"* ]]; then 
        conv_mode="llama2_harm2"
    elif [[ $MODEL_PATH == *"redteam"* ]]; then  
        conv_mode="llama2_harm"
    else 
        conv_mode="llama2"
    fi
version="-woffn"


echo "Processing $domain"
mkdir -p ${LOGS_BASE_PATH}/${domain}

# 执行 infer 命令
srun -p partition --gres=gpu:1  --quotatype=auto --output=${LOGS_BASE_PATH}/${domain}/${CKPT}${version}.infer.log python -m ming.eval.model_advbench_gen \
    --model-base ${MODEL_BASE} \
    --model-path ${MODEL_PATH} \
    --answers-file ${LOGS_BASE_PATH}/${domain}/${CKPT}${version}.jsonl \
    --s3-answers-file s3://syj_test/logs/diverse/${domain}/${CKPT}${version}.jsonl \
    --question-file /mnt/petrelfs/jiangshuyang.p/datasets/harmful_behaviors.csv \
    --temperature 0 \
    --max-tokens 1024 \
    --keep-local \
    --conv-mode ${conv_mode} \
    --infer-answer \
    --use-logit-bias \
    --unload-ffn \
    --resume &

wait 

echo "Evaluating $domain"

srun -p partition python -m ming.eval.eval_advbench --input_file ${LOGS_BASE_PATH}/${domain}/${CKPT}${version}.jsonl
