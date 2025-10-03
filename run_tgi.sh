#!/bin/bash
#
# Expects 1 argument that specifies the model to load with TGI. For example,
#   ./run_tgi.sh aviation_Llama-2-13b-chat-hf_v2
#   ./run_tgi.sh aviation_Llama-2-7b-chat-hf_v2
#   ./run_tgi.sh aviation_Mistral-7B-v0.1_v3
#   ./run_tgi.sh Llama-2-13b-chat-hf
#   ./run_tgi.sh Llama-2-7b-chat-hf
#   ./run_tgi.sh Mistral-7B-Instruct-v0.1
#   ./run_tgi.sh Mistral-7B-Instruct-v0.2

volume=/projects/alue/models
model_id=/data/$1/
tgi_version=2.1.1
num_shard="4" # Can change to 2, 3, 4, etc. based on number of GPUs
max_input_length=8000
max_batch_prefill_tokens=$max_input_length
max_total_tokens=8192

# docker kill tgi
docker run -d --name $1 --gpus all --shm-size 1g -p 3000:80 -v $volume:/data --rm ghcr.io/huggingface/text-generation-inference:$tgi_version --model-id $model_id --huggingface-hub-cache /data --num-shard $num_shard --max-input-length $max_input_length --max-batch-prefill-tokens $max_batch_prefill_tokens --max-total-tokens $max_total_tokens
