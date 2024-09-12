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

from slicegpt import gpu_utils, hf_utils, utils
from slicegpt.config import config

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
        orig_parameters = sum(p.numel() for p in model.parameters())

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

        new_parameters = sum(p.numel() for p in model.parameters())
        print ("Sparsity achieved {}%".format(int(100*(1-new_parameters/orig_parameters))))

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

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size)[
        'results'
    ]

    logging.info(results)
    wandb.log(results)

    with open(f"{args.save_dir}/full_results_{args.num_fewshot}_shot.json", "w") as f:
        json.dump(results, f)

    metric_vals = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    acc_avg = calculate_avg_accuracy(task_names, results)
    metric_vals['average'] = round(acc_avg, 4)
    with open(f"{args.save_dir}/{args.num_fewshot}_shot_task_results.json", "w") as f:
        json.dump(metric_vals, f)

    wandb.log({'acc_avg': acc_avg})

    logging.info(json.dumps(metric_vals, indent=4))
    logging.info(f"Average accuracy across tasks: {acc_avg}")


if __name__ == "__main__":
    # Use the logger from lm_eval, adding a file handler to write the log to file
    logging = lm_eval_utils.eval_logger
    logging.addHandler(utils.create_file_handler(log_dir="log"))

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    eval_args = eval_arg_parser()
    process_eval_args(eval_args)
    eval_main(eval_args)
