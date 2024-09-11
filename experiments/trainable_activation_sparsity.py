import logging

logging.basicConfig()
logger = logging.getLogger(__name__)

import argparse
from datasets import load_dataset, DatasetDict
from evaluate import load
from functools import partial
from itertools import product
import json
import pandas as pd
from pathlib import Path
import random
import statistics
from safetensors.torch import load_file
import time
import torch
from tqdm import tqdm
from copy import deepcopy as cp
import numpy as np

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from peft import LoraConfig, TaskType, get_peft_model

from bo_options import lora_target_map
from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions import Categorical, Bernoulli
from torch.nn.utils.parametrizations import orthogonal


class SparsityPredictor(torch.nn.Module):
    def __init__(
        self, hidden_size=768, intermediate_size=3072, sparsity_level=0.0
    ):
        super(SparsityPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # self.proj_hidden = orthogonal(nn.Linear(hidden_size, hidden_size))  # (768, 768)
        #self.proj_intermediate = orthogonal(
        #    nn.Linear(hidden_size, intermediate_size)
        #)  # (3072, 768)
        self.proj_intermediate = nn.Linear(hidden_size, intermediate_size, bias=False)

        #self.proj_intermediate2 = nn.Linear(intermediate_size, intermediate_size)  # (3072, 768)

        self.row_sparsities = nn.Parameter(
            torch.rand(intermediate_size, 1), requires_grad=True
        )  # (3072, 1)
        # self.col_sparsities = nn.Parameter(torch.rand(hidden_size, 1), requires_grad=True)  # (768, 1)
        self.sparsity_level = sparsity_level

        self.singular_value = None
        #for param in self.proj_intermediate.parameters():
        #    param.requires_grad = False

    def calculate_KLD(self):
        return (
            -1 * torch.log(self.alpha) * (1 - self.alpha)
            - self.alpha * torch.log(1 - self.alpha)
            + (torch.tensor(1-self.sparsity_level) * torch.log(torch.tensor(self.sparsity_level))).to(self.alpha.device)
            + (torch.tensor(self.sparsity_level) * torch.log(torch.tensor(1 - self.sparsity_level))).to(self.alpha.device)
        ).sum()

    def calculate_l1_loss(self):
        return torch.sum(torch.abs(self.keep_probs - self.sparsity_level))

    def calculate_total_loss(self):
        return self.calculate_KLD()  # + self.calculate_l1_loss()

    def forward(self, weight_matrix):
        # if weight_matrix.shape[0] == self.hidden_size:
        #     proj_ = self.proj_hidden(weight_matrix.T)
        #     alpha = nn.Sigmoid()(proj_ @ self.col_sparsities)[:,0]
        #print (weight_matrix)
        #print (self.proj_intermediate.weight.data)

        if weight_matrix.shape[0] == self.intermediate_size:  # (3072, 768)
            proj_ = self.proj_intermediate(weight_matrix)  # (3072, 3072)
            _, s, _ = torch.svd(cp(self.proj_intermediate.weight.data).to(torch.float32))
            #self.singular_value = s
            proj_ = proj_/s.max()
            proj_ = nn.ReLU()(proj_)
            #proj_ = self.proj_intermediate2(proj_)
            alpha = nn.Sigmoid()(proj_ @ self.row_sparsities)[:, 0]  # (3072, )
        else:
            raise ValueError("The layer does not support sparsity operation")

        self.alpha = alpha

        m = Uniform(torch.tensor([self.sparsity_level]), torch.tensor([1.0]))
        eps = m.sample((alpha.shape[0],)).to(weight_matrix.device)[
            :, 0
        ]  # (3072, )

        # Calculate the probabilities using reparametrization trick
        keep_probs = nn.Sigmoid()(
            torch.log(eps)
            - torch.log(1 - eps)
            + torch.log(alpha)
            - torch.log(1 - alpha)
        )

        self.keep_probs = keep_probs

        # Use the keep_probs as a mask to determine which rows to keep
        # rows_to_keep = keep_probs <= 0.5

        return keep_probs


def calculate_activation_reward(weight_matrix):
    if weight_matrix.dtype == torch.float16:
        weight_matrix = weight_matrix.to(torch.float32)

    u,s,v = torch.svd(weight_matrix)
    x = torch.abs(s.max()-1)
    x += 0.0001
    #print ("Reward")
    #print (weight_matrix, 1/x)
    return 1/x

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def get_optimizer_and_scheduler(model, train_dataset, config):
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    kwargs_lr_scheduler = {
        "optimizer": optimizer,
        "num_warmup_steps": config.num_warmup_steps,
        "num_training_steps": (
            (len(train_dataset) - 1) // (config.finetune_train_batch_size * config.gradient_accumulation_steps) + 1
        )
        * config.epochs,
    }
    if config.lr_scheduler_type in ("cosine", "cosine_with_warmup"):
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(**kwargs_lr_scheduler)
    elif config.lr_scheduler_type in ("linear", "linear_with_warmup"):
        lr_scheduler = transformers.get_linear_schedule_with_warmup(**kwargs_lr_scheduler)
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler


class CustomTrainer(Trainer):
    def __init__(self, *args, train_loader=None, test_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def get_eval_dataloader(self, _) -> DataLoader:
        return self.test_loader

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def finetune(model):
    # get the dataset for finetuning
    finetune_ds = data_utils.get_dataset(args.finetune_dataset)
    finetune_train_loader = data_utils.prepare_dataloader(
        dataset=finetune_ds["train"],
        tokenizer=tokenizer,
        max_seqlen=args.finetune_train_seqlen,
        batch_size=args.finetune_train_batch_size,
        nsamples=args.finetune_train_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    finetune_test_loader = data_utils.prepare_dataloader(
        dataset=finetune_ds["test"],
        tokenizer=tokenizer,
        max_seqlen=args.finetune_test_seqlen,
        batch_size=args.finetune_test_batch_size,
        nsamples=args.finetune_test_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_target_map(args.model_name)[args.lora_target_option],
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    # create optimizer and scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(lora_model, finetune_ds["train"], args)

    training_args = TrainingArguments(
        output_dir=args.st_checkpoint_dir,  # output directory
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.finetune_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.finetune_test_batch_size,  # batch size for evaluation
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        disable_tqdm=False,
        load_best_model_at_end=True,
        eval_steps=args.eval_steps,
        evaluation_strategy=args.evaluation_strategy,
        report_to='tensorboard',
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # lower eval_loss is better,
        gradient_checkpointing=True,
    )

    trainer = CustomTrainer(
        model=lora_model,
        tokenizer=tokenizer,
        train_loader=finetune_train_loader,
        test_loader=finetune_test_loader,
        args=training_args,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # required to enable gradient_checkpointing
    lora_model.enable_input_require_grads()

    lora_model.train()
    trainer.train()

    return lora_model

def _get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        dest="loglevel",
        default="DEBUG",
        help="Set the log level for the logging module.",
    )
    parser.add_argument(
        "--use_gpu",
        dest="use_gpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use the gpu or not.",
    )

    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    
    parser.add_argument(
        "--model_name",
        dest="model_name",
        default="",
        help="Name of the model checkpoint.",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        default="",
        help="The path containing the model checkpoint.",
    )
    parser.add_argument(
        "--num_episodes",
        dest="num_episodes",
        default=20,
        help="Number of episodes to train model.",
    )

    parser.add_argument(
        "--sparsity_level",
        dest="sparsity_level",
        default=0.2,
        help="Sparsity level of the model.",
    )

    parser.add_argument(
        "--learning-rate-action",
        dest="learning_rate_action",
        default=0.001,
        help="Learning rate to train sparsity model.",
    )

    # Perplexity evaluation command-line arguments
    parser.add_argument(
        "--ppl-eval-dataset",
        type=str,
        help="Dataset to evaluate perplexity.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--ppl-eval-nsamples",
        type=int,
        help="Number of samples of the perplexity eval dataset to load.",
        default=128,
    )
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )

    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")

    parser.add_argument("--finetune", action="store_true", help="Whether to finetune model.")

    # finetuning command-line arguments
    parser.add_argument(
        "--finetune-dataset",
        type=str,
        help="Dataset to finetune on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--finetune-train-nsamples",
        type=int,
        help="Number of samples to load from the train set for finetuning.",
        default=4096,
    )
    parser.add_argument(
        "--finetune-test-nsamples",
        type=int,
        help="Number of samples to load from the test set for finetuning.",
        default=128,
    )
    parser.add_argument("--finetune-train-batch-size", type=int, default=1, help="Batch size for finetuning training.")
    parser.add_argument("--finetune-test-batch-size", type=int, default=8, help="Batch size for finetuning testing.")
    parser.add_argument(
        "--finetune-train-seqlen", type=int, default=2048, help="Sequence length for finetuning training."
    )
    parser.add_argument(
        "--finetune-test-seqlen", type=int, default=2048, help="Sequence length for finetuning testing."
    )

    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.95)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--lr-scheduler-type', type=str, default="linear")
    parser.add_argument('--num-warmup-steps', type=int, default=400)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--early-stopping-patience', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--evaluation-strategy', type=str, default="steps")
    parser.add_argument('--eval-steps', type=int, default=16)
    parser.add_argument('--save-steps', type=int, default=16)
    parser.add_argument('--save-total-limit', type=int, default=1)
    parser.add_argument('--logging-steps', type=int, default=1)

    parser.add_argument('--lora-alpha', type=float, default=32.0)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-bias', type=str, default="none")

    parser.add_argument(
        '--st_checkpoint_dir', type=str, default=".", help="Path for syne-tune to save finetuning checkpoints."
    )
    parser.add_argument(
        '--lora-target-option',
        default="attn_head_and_mlp",
        help="target module option to apply lora to (names of attn i/p, attn o/p and mlp in LayerAdapter)",
    )

    parser.add_argument('--wandb-project', type=str, default="benoulligpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_parse_args()

    args.num_episodes = int(args.num_episodes)
    args.sparsity_level = float(args.sparsity_level)
    args.learning_rate_action = float(args.learning_rate_action)

    logger.setLevel(getattr(logging, args.loglevel.upper()))
    logger.debug(f"args: {args}")

    # otherwise, continue with the experiments
    # set random seed
    set_seed(args.seed)

    device = "cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu"

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")

    #config = AutoConfig.from_pretrained(args.model_name)

    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model_name)

    #tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # load the pretrained model with the config
    #model_orig = AutoModelForCausalLM.from_config(config).to(device)

    model_orig = cp(model_adapter.model).to(device)

    # get the dataset for perplexity evaluation
    ppl_ds = data_utils.get_dataset(args.ppl_eval_dataset)
    ppl_eval_loader = data_utils.prepare_dataloader(
        dataset=ppl_ds["validation"],
        tokenizer=tokenizer,
        max_seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size,
        nsamples=args.ppl_eval_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    if args.finetune:
        model_orig = finetune(model_orig)

    model_orig.eval()

    dataset_ppl = gpu_utils.evaluate_ppl(model_orig, model_orig.config.pad_token_id, ppl_eval_loader)
    print(f'PPL before finetuning: {dataset_ppl:.4f}')
    #wandb.log({"pre_finetune_ppl": dataset_ppl})

    utils.cleanup_memory()

    orig_parameters = sum(p.numel() for p in model_orig.parameters())

    all_rewards = []
    best_score = 0

    try:
        action_model = SparsityPredictor(
            model_orig.config.hidden_size, model_orig.config.intermediate_size, args.sparsity_level
        )
    except:
        action_model = SparsityPredictor(
            model_orig.config.hidden_size, model_orig.config.ffn_dim, args.sparsity_level
        )

    #if args.dtype == "fp16":
    #    action_model = action_model.half()

    action_model.to(device)

    action_model.train()

    print(action_model)

    print(
        "Total number of parameters",
        sum(p.numel() for p in action_model.parameters() if p.requires_grad),
    )

    optimizer = torch.optim.AdamW(
        action_model.parameters(), lr=args.learning_rate_action
    )

    best_accuracy = 0

    scaler = torch.cuda.amp.GradScaler()

    for episode in tqdm(range(args.num_episodes)):
        #optimizer.zero_grad()

        state_pool = []
        action_pool = []
        reward_pool = []

        model = cp(model_orig)
        #model_name, main_model = list(model.named_modules())[1]

        total_loss = 0
        count = 0
        total_reward = 0

        for layer in model.base_model.model.model.decoder.layers:
            weight = layer.fc1.weight.data  # (3072, 768)
            state = Variable(cp(weight))
            
            # print (weight)

            with torch.autocast(device_type=device, dtype=torch.float16):
                o = action_model(state)  # (3072, )
                #print (state)
                #for name, param in action_model.named_parameters():
                #    print (name)
                #    print (param)

                y = Bernoulli(o).sample()
                #y = (o > args.sparsity_level).long().to(o.device)  # (3072, )
                row_indices = (y != 0).nonzero()[:, 0]  # len less than 3072

            # slice the intermediate and output weight matrices appropriately
            layer.fc1.out_features = len(row_indices)
            layer.fc1.weight.data = (
                layer.fc1.weight[row_indices, :]
            )

            #try:
            layer.fc1.lora_B.default.weight.data = (
                layer.fc1.lora_B.default.weight[row_indices, :]
            )
            #except:
            #    pass

            layer.fc1.bias.data = layer.fc1.bias[
                row_indices
            ]

            # revert changes on output layer
            layer.fc2.in_features = len(row_indices)
            layer.fc2.weight.data = layer.fc2.weight[
                :, row_indices
            ]

            #try:
            layer.fc2.lora_A.default.weight.data = (
                layer.fc2.lora_A.default.weight[:, row_indices]
            )
            #except:
            #    pass

            # print(layer)      # matrices are indeed being sliced, verified from the output

            # get the updated rewards
            reward = calculate_activation_reward(layer.fc1.weight.data)

            state_pool.append(state)
            action_pool.append(y)
            reward_pool.append(reward.item())

            total_reward += reward.item()
            count += 1

            #print (reward)
            
        reward_pool = discount_rewards(reward_pool)
        new_parameters = sum(p.numel() for p in model.parameters())

        for i in range(len(state_pool)):
            with torch.autocast(device_type=device, dtype=torch.float16):
                state = state_pool[i]
                action = Variable(action_pool[i])
                reward = reward_pool[i]

                o = action_model(state)  # (3072, )
                y = Bernoulli(o)

                loss = -y.log_prob(action).sum() * reward  # Negtive score function x reward
            
            #loss.backward()
            #print (loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            #torch.nn.utils.clip_grad_norm_(action_model.parameters(), 1.0)
        
        #optimizer.step()
        #optimizer.zero_grad()

        #for name, param in action_model.named_parameters():
        #    print (name, param)

        state_pool = []
        action_pool = []
        reward_pool = []

        dataset_ppl2 = gpu_utils.evaluate_ppl(model, model.config.pad_token_id, ppl_eval_loader)
        print(f'PPL after sparsification: {dataset_ppl2:.4f}')
        #wandb.log({"post_sparsification_ppl": dataset_ppl2})

        print(
            "Episode",
            episode,
            "Avg reward",
            total_reward/count,
            "sparsity",
            str((1-new_parameters/orig_parameters)*100) + " %",
            "Change in ppl",
            dataset_ppl2-dataset_ppl
        )

        #if total_reward > best_score:
        #    best_score =  total_reward
        #    torch.save(action_model.state_dict(), "action_model.ckpt")
        #    #torch.save(model.state_dict(), "sliced_model.bin")

# TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python trainable_activation_sparsity.py --log DEBUG --use_gpu --model_name facebook/opt-125m  --num_episodes 100 --learning-rate-action 0.01 --sparsity_level 0.2 --ppl-eval-dataset wikitext2            --finetune-dataset wikitext2            --finetune-train-nsamples 8000            --finetune-train-seqlen 1024            --finetune-train-batch-size 3            --lora-alpha 10            --lora-r 32            --lora-dropout 0.05            --lora-target-option attn_head_and_mlp            --eval-steps 16            --save-steps 16 --finetune --no-wandb --epochs 3