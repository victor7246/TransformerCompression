for model in facebook/opt-125m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b
do

#TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1  python run_lm_eval.py  \
#    --model ${model}    \
#    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\

for dataset in wikitext2 alpaca
do

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1  python run_lm_eval.py  \
    --model ${model}    \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\
    --finetune \
    --finetune-dataset ${dataset} \

done
done

for model in facebook/opt-125m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b
do
for sparsity_level in 0.1 0.2 0.3 0.4 0.5
do
for sparsity_technique in random
do

#TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1  python run_lm_eval.py  \
#    --model ${model}    \
#    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\
#    --sparsity ${sparsity_level} \
#    --sparsity_technique ${sparsity_technique} \
#    --use-slicing

for dataset in wikitext2 alpaca
do
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1  python run_lm_eval.py  \
    --model ${model}    \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\
    --sparsity ${sparsity_level} \
    --sparsity_technique ${sparsity_technique} \
    --use-slicing \
    --finetune \
    --finetune-dataset ${dataset}
done
done

for sparsity_technique in bernoulli
do

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1  python run_lm_eval.py  \
    --model ${model}    \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\
    --sparsity ${sparsity_level} \
    --sparsity_technique ${sparsity_technique} \
    --use-slicing \
    --no-wandb

for dataset in wikitext2 alpaca
do
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1  python run_lm_eval.py  \
    --model ${model}    \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\
    --sparsity ${sparsity_level} \
    --sparsity_technique ${sparsity_technique} \
    --use-slicing \
    --finetune \
    --finetune-dataset ${dataset} \
    --no-wandb
done
done
done
done

================ Siddhant Changes ========================
# slice with RL model
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2  python run_lm_eval.py  \
    --model facebook/opt-125m     \
    --sparsity 0.2        \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa      \
    --no-wandb  \
    --use-slicing \
    --slice_with_action_model \
    --finetune

# slice with uniform sampling
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2  python run_lm_eval.py  \
    --model facebook/opt-125m     \
    --sparsity 0.2        \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa      \
    --no-wandb  \
    --use-slicing \
    --slice_uniform \
    --finetune

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2  python run_lm_eval.py  \
    --model facebook/opt-125m     \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa      \
    --no-wandb  \
    --finetune
