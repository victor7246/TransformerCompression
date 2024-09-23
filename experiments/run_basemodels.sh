#for model in meta-llama/Llama-2-7b-hf microsoft/phi-2 facebook/opt-6.7b facebook/opt-2.7b facebook/opt-1.3b facebook/opt-125m 
#do
#for dataset in wikitext2 #alpaca
#do

#TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python finetune_basemodel.py \
#    --log DEBUG \
#    --use_gpu \
#    --model_name ${model}  \
#    --ppl-eval-dataset ${dataset}       \
#    --finetune-dataset ${dataset}         \
#    --finetune-train-nsamples 8000       \
#    --finetune-train-seqlen 1024       \
#    --finetune-train-batch-size 3         \
#    --lora-alpha 10          \
#    --lora-r 32        \
#    --lora-dropout 0.05      \
#    --lora-target-option attn_head_and_mlp      \
#    --eval-steps 16       \
#    --save-steps 16 \
#    --epochs 1 \
#    --model_save_path "../models/" 

#done
#done

for model in microsoft/phi-2
do
for dataset in wikitext2 #alpaca
do
for activation in 'relu' 'leakygelu'
do

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python finetune_basemodel.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
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
    --activation ${activation}
done
done
done