for model in meta-llama/Llama-2-7b-hf
do
for sparsity_level in 0.1 0.2 0.25 0.3
do
for sparsity_technique in bernoulli
do

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=0 CUDA_VISIBLE_DEVICES=1 python trainable_activation_sparsity_allmodules.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 10\
    --learning-rate-action 0.0005 \
    --sparsity_level ${sparsity_level} \
    --model_save_path "../models2/" \
    --sparsity_technique ${sparsity_technique} \
    --no-wandb

done  
done
done