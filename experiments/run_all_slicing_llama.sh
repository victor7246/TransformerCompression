for model in meta-llama/Llama-2-7b-hf
do
for dataset in wikitext2 alpaca
do
for sparsity_level in 0.1 0.2 0.3 0.4 0.5
do
for sparsity_technique in random bernoulli
do
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python trainable_activation_sparsity.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 20 \
    --learning-rate-action 0.0005 \
    --sparsity_level ${sparsity_level} \
    --ppl-eval-dataset ${dataset}       \
    --finetune-dataset ${dataset}         \
    --finetune-train-nsamples 8000       \
    --finetune-train-seqlen 1024       \
    --finetune-train-batch-size 3         \
    --lora-alpha 10          \
    --lora-r 32        \
    --lora-dropout 0.05      \
    --lora-target-option attn_head_and_mlp      \
    --eval-steps 16       \
    --save-steps 16 \
    --epochs 1 \
    --model_save_path "../models/" \
    --sparsity_technique ${sparsity_technique}

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python trainable_activation_sparsity.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 20 \
    --learning-rate-action 0.0005 \
    --sparsity_level ${sparsity_level} \
    --ppl-eval-dataset ${dataset}       \
    --finetune-dataset ${dataset}         \
    --finetune-train-nsamples 8000       \
    --finetune-train-seqlen 1024       \
    --finetune-train-batch-size 3         \
    --lora-alpha 10          \
    --lora-r 32        \
    --lora-dropout 0.05      \
    --lora-target-option attn_head_and_mlp      \
    --eval-steps 16       \
    --save-steps 16 \
    --epochs 1 \
    --model_save_path "../models/" \
    --sparsity_technique ${sparsity_technique} \
    --activation "leakysilu"

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python trainable_activation_sparsity.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 20 \
    --learning-rate-action 0.0005 \
    --sparsity_level ${sparsity_level} \
    --ppl-eval-dataset ${dataset}       \
    --finetune-dataset ${dataset}         \
    --finetune-train-nsamples 8000       \
    --finetune-train-seqlen 1024       \
    --finetune-train-batch-size 3         \
    --lora-alpha 10          \
    --lora-r 32        \
    --lora-dropout 0.05      \
    --lora-target-option attn_head_and_mlp      \
    --eval-steps 16       \
    --save-steps 16 \
    --epochs 1 \
    --model_save_path "../models/" \
    --sparsity_technique ${sparsity_technique} \
    --activation "relu"

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python trainable_activation_sparsity.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 20 \
    --learning-rate-action 0.0005 \
    --sparsity_level ${sparsity_level} \
    --ppl-eval-dataset ${dataset}       \
    --finetune-dataset ${dataset}         \
    --finetune-train-nsamples 8000       \
    --finetune-train-seqlen 1024       \
    --finetune-train-batch-size 3         \
    --lora-alpha 10          \
    --lora-r 32        \
    --lora-dropout 0.05      \
    --lora-target-option attn_head_and_mlp      \
    --eval-steps 16       \
    --save-steps 16 \
    --epochs 1 \
    --model_save_path "../models/" \
    --finetune \
    --sparsity_technique ${sparsity_technique}

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python trainable_activation_sparsity.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 20 \
    --learning-rate-action 0.0005 \
    --sparsity_level ${sparsity_level} \
    --ppl-eval-dataset ${dataset}       \
    --finetune-dataset ${dataset}         \
    --finetune-train-nsamples 8000       \
    --finetune-train-seqlen 1024       \
    --finetune-train-batch-size 3         \
    --lora-alpha 10          \
    --lora-r 32        \
    --lora-dropout 0.05      \
    --lora-target-option attn_head_and_mlp      \
    --eval-steps 16       \
    --save-steps 16 \
    --epochs 1 \
    --model_save_path "../models/" \
    --finetune \
    --sparsity_technique ${sparsity_technique} \
    --activation "leakysilu"

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python trainable_activation_sparsity.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 20 \
    --learning-rate-action 0.0005 \
    --sparsity_level ${sparsity_level} \
    --ppl-eval-dataset ${dataset}       \
    --finetune-dataset ${dataset}         \
    --finetune-train-nsamples 8000       \
    --finetune-train-seqlen 1024       \
    --finetune-train-batch-size 3         \
    --lora-alpha 10          \
    --lora-r 32        \
    --lora-dropout 0.05      \
    --lora-target-option attn_head_and_mlp      \
    --eval-steps 16       \
    --save-steps 16 \
    --epochs 1 \
    --model_save_path "../models/" \
    --finetune \
    --sparsity_technique ${sparsity_technique} \
    --activation "relu"
    
done  
done
done
done