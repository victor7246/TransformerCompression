TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2  python run_lm_eval.py  \
    --model facebook/opt-125m     \
    --sparsity 0.2        \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa      \
    --no-wandb  \
    --use-slicing \
    --finetune

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2  python run_lm_eval.py  \
    --model facebook/opt-125m     \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa      \
    --no-wandb  \
    --finetune