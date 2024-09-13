# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os

import lm_eval
import torch
import wandb
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from bo_options import lora_target_map
from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from peft import LoraConfig, TaskType, get_peft_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions import Categorical, Bernoulli
from torch.nn.utils.parametrizations import orthogonal

from copy import deepcopy as cp

class SparsityPredictor(torch.nn.Module):
    def __init__(
        self, hidden_size=768, intermediate_size=3072, sparsity_level=0.2
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
        self.row_sparsities_bias = nn.Parameter(
            torch.rand(1, 1), requires_grad=True
        )  # (3072, 1)
        # self.col_sparsities = nn.Parameter(torch.rand(hidden_size, 1), requires_grad=True)  # (768, 1)
        self.sparsity_level = sparsity_level
        self.density_level = 1-sparsity_level

        self.singular_value = None
        #for param in self.proj_intermediate.parameters():
        #    param.requires_grad = False

    def calculate_KLD(self):
        return (
            -1 * torch.log(self.alpha) * (1 - self.alpha)
            - self.alpha * torch.log(1 - self.alpha)
            + (torch.tensor(self.density_level) * torch.log(torch.tensor(self.sparsity_level))).to(self.alpha.device)
            + (torch.tensor(self.sparsity_level) * torch.log(torch.tensor(self.density_level))).to(self.alpha.device)
        ).sum()

    def calculate_l1_loss(self):
        return torch.sum(torch.abs(self.keep_probs - self.density_level))
        #return torch.abs((self.keep_probs > 0.5).sum()/self.keep_probs.shape[0] - self.density_level)

    def calculate_total_loss(self):
        return self.calculate_KLD()

    def forward(self, weight_matrix):
        # if weight_matrix.shape[0] == self.hidden_size:
        #     proj_ = self.proj_hidden(weight_matrix.T)
        #     alpha = nn.Sigmoid()(proj_ @ self.col_sparsities)[:,0]
        #print (weight_matrix)
        #print (self.proj_intermediate.weight.data)

        if weight_matrix.shape[0] == self.intermediate_size:  # (3072, 768)
            proj_ = self.proj_intermediate(weight_matrix)  # (3072, 3072)
            #_, s, _ = torch.svd(cp(self.proj_intermediate.weight.data).to(torch.float32))
            #self.singular_value = s
            #proj_ = proj_/s.max()
            #proj_ = nn.ReLU()(proj_)
            #proj_ = self.proj_intermediate2(proj_)
            alpha = nn.Sigmoid()(proj_ @ self.row_sparsities + self.row_sparsities_bias)[:, 0]  # (3072, )
        else:
            raise ValueError("The layer does not support sparsity operation")

        self.alpha = alpha

        m = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
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

        keep_probs = torch.clip(keep_probs, max=self.density_level)

        self.keep_probs = keep_probs

        # Use the keep_probs as a mask to determine which rows to keep
        # rows_to_keep = keep_probs <= 0.5

        return keep_probs

TASK_METRIC_MAP = {
    "mmlu_abstract_algebra": "acc,none",
    "mmlu_business_ethics": "acc,none",
    "mmlu_college_computer_science": "acc,none",
    "mmlu_college_mathematics": "acc,none",
    "mmlu_conceptual_physics": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_machine_learning": "acc,none",
    "mmlu_miscellaneous": "acc,none",
    "mmlu_philosophy": "acc,none",
    "mmlu_global_facts": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}

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
    
def eval_arg_parser(interactive: bool = True) -> argparse.Namespace:
    initialize_tasks()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument('--use-slicing', action="store_true", help="Use slicing.")
    parser.add_argument('--slice_uniform', default=False, action="store_true", help="Whether to slice rows uniformly at random using the provided sparsity level.")
    parser.add_argument('--slice_with_action_model', default = False, action="store_true", help="Whether to use the trained action model to slice rows.")
    parser.add_argument('--finetune', action="store_true", help="Fine tune model.")

    parser.add_argument('--wandb-project', type=str, default="slicegpt-lm-eval", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--save-dir", type=str, default=".", help="Path to save the lm eval results")

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

    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")

    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    
    parser.add_argument("--use-kld", action="store_true", help="To use KLD loss")

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

    return parser.parse_args() if interactive else parser.parse_args('')


def process_eval_args(args: argparse.Namespace):
    logging.info(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.info(f'{arg} = {argv}')


def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if 'mmlu' not in task)

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get(TASK_METRIC_MAP[task]) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())
    wandb.log({'acc_mmlu_avg': acc_mmlu_avg})

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)

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

def finetune(model, skip_lora=False):
    # get the dataset for finetuning
    _, tokenizer = hf_utils.get_model_and_tokenizer(args.model)
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

    if skip_lora == False:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_target_map(args.model)[args.lora_target_option],
        )

        lora_model = get_peft_model(model, lora_config)
    else:
        lora_model = model

    if args.finetune:
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

def _slice_uniformly(model, args):
    for layer in model.base_model.decoder.layers:
        # sample a bunch of rows uniformly at random
        num_rows = layer.fc1.out_features
        row_indices = torch.randperm(num_rows)[int(num_rows * args.sparsity): ].sort().values

        # slice
        layer.fc1.out_features = len(row_indices)
        layer.fc1.weight.data = (
            layer.fc1.weight[row_indices, :]
        )

        try:
            layer.fc1.lora_B.default.weight.data = (
                layer.fc1.lora_B.default.weight[row_indices, :]
            )
        except:
            pass

        layer.fc1.bias.data = layer.fc1.bias[
            row_indices
        ]

        # revert changes on output layer
        layer.fc2.in_features = len(row_indices)
        layer.fc2.weight.data = layer.fc2.weight[
            :, row_indices
        ]

        try:
            layer.fc2.lora_A.default.weight.data = (
                layer.fc2.lora_A.default.weight[:, row_indices]
            )
        except:
            pass



def _slice_with_action_model(model, action_model):
    for layer in model.base_model.decoder.layers:
        weight = layer.fc1.weight.data  # (3072, 768)
        state = Variable(cp(weight))
        
        # print (weight)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                o = action_model(state)  # (3072, )
                y = Bernoulli(o).sample()
                #y = (o > args.sparsity_level).long().to(o.device)  # (3072, )
                row_indices = (y != 0).nonzero()[:, 0]  # len less than 3072

        # slice the intermediate and output weight matrices appropriately
        layer.fc1.out_features = len(row_indices)
        layer.fc1.weight.data = (
            layer.fc1.weight[row_indices, :]
        )

        try:
            layer.fc1.lora_B.default.weight.data = (
                layer.fc1.lora_B.default.weight[row_indices, :]
            )
        except:
            pass

        layer.fc1.bias.data = layer.fc1.bias[
            row_indices
        ]

        # revert changes on output layer
        layer.fc2.in_features = len(row_indices)
        layer.fc2.weight.data = layer.fc2.weight[
            :, row_indices
        ]

        try:
            layer.fc2.lora_A.default.weight.data = (
                layer.fc2.lora_A.default.weight[:, row_indices]
            )
        except:
            pass

def eval_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT LM eval experiment.")

    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    if args.sliced_model_path:
        # load the sliced model
        logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            token=args.hf_token,
            round_interval=args.round_interval,
        )
    else:
        # load the original model
        logging.info(f"Loading {args.model} model")
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, args.model_path, token=args.hf_token)

    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    model_adapter.model.tie_weights = lambda: None

    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(config.device)

    ### LM Eval Harness ###

    model = cp(model_adapter.model)

    if args.use_slicing:
        # either use uniform slicing or slicing with action model
        assert args.slice_with_action_model or args.slice_uniform

        orig_parameters = sum(p.numel() for p in model.parameters())

        if args.slice_with_action_model:
            try:
                action_model = SparsityPredictor(
                    model.config.hidden_size, model.config.intermediate_size, args.sparsity
                )
            except:
                action_model = SparsityPredictor(
                    model.config.hidden_size, model.config.ffn_dim, args.sparsity
                )
            
            action_model.to(config.device)
            
            action_model.load_state_dict(torch.load("action_model.ckpt"))
            action_model.eval()

            # do the slicing
            _slice_with_action_model(model, action_model)
        else:
            # must be slicing uniformly
            _slice_uniformly(model, args)

        new_parameters = sum(p.numel() for p in model.parameters())
        print ("Sparsity achieved {}%".format(int(100*(1-new_parameters/orig_parameters))))

    if args.finetune:
        model = finetune(model)

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    logging.info(f"Selected Tasks: {task_names}")

    for task in task_names:
        if task not in TASK_METRIC_MAP:
            raise NotImplementedError(
                f"Please specify the metric to use for {task} in TASK_METRIC_MAP. Available info {TASK_METRIC_MAP}"
            )

    # results is a dict with keys: results, configs, versions, n-shot, samples, config, git_hash
    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size)[
        'results'
    ]

    logging.info(results)
    wandb.log(results)

    # name the output file
    outfile = f"{args.save_dir}/full_results_{args.num_fewshot}_shot"
    if args.use_slicing:
        if args.slice_with_action_model:
            outfile += "_actionModelSlicing"
        else:
            outfile += "_uniformSlicing"
    outfile += ".json"

    with open(outfile, "w") as f:
        json.dump(results, f, indent=4)

    metric_vals = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    acc_avg = calculate_avg_accuracy(task_names, results)
    metric_vals['average'] = round(acc_avg, 4)

    # save this in the task results output file
    task_outfile = f"{args.save_dir}/{args.num_fewshot}_shot_task_results"
    if args.use_slicing:
        if args.slice_with_action_model:
            task_outfile += "_actionModelSlicing"
        else:
            task_outfile += "_uniformSlicing"
    task_outfile += ".json"

    with open(task_outfile, "w") as f:
        json.dump(metric_vals, f, indent=4)

    wandb.log({'acc_avg': acc_avg})

    logging.info(json.dumps(metric_vals, indent=4))
    logging.info(f"Average accuracy across tasks: {acc_avg}")


if __name__ == "__main__":
    # Use the logger from lm_eval, adding a file handler to write the log to file
    logging = lm_eval_utils.eval_logger
    logging.addHandler(utils.create_file_handler(log_dir="log"))

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    args = eval_arg_parser()
    process_eval_args(args)
    eval_main(args)
