for model in meta-llama/Llama-2-7b-hf
do
for sparsity in 0.1 0.2 0.25 0.3
do
for sparsity_technique in bernoulli
do
for datasize in 100 1000 2000
do
for dataset in wikitext2 #alpaca
do
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python run_lm_eval.py  \
    --model ${model}    \
    --tasks hellaswag arc_easy arc_challenge winogrande piqa mmlu_abstract_algebra mmlu_business_ethics mmlu_college_computer_science mmlu_college_mathematics mmlu_conceptual_physics mmlu_formal_logic mmlu_machine_learning mmlu_miscellaneous mmlu_philosophy mmlu_global_facts\
    --sparsity ${sparsity} \
    --sparsity_technique ${sparsity_technique} \
    --use-slicing \
    --finetune \
    --finetune-dataset ${dataset} \
    --finetune-train-nsamples ${datasize}

done
done
done
done
done
