import os
import time
import torch
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    LoraModel
)
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm


# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
parser.add_argument("--dataset", type=str, default="cola")
parser.add_argument("--task", type=str, default="cola")
parser.add_argument("--peft", type=str, default="kasa")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.0)
parser.add_argument("--head_lr", type=float, default=4e-4)
parser.add_argument("--module_lr", type=float, default=4e-4)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warmup_ratio", type=float, default=0.06)
# parser.add_argument("--train_ratio", type=float, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--beta", type=float, default=1e-4)
parser.add_argument("--gemma", type=float, default=1e-3)
args = parser.parse_args()
for arg, value in vars(args).items():
    print(f'{arg}: {value}')

# logging
def log(*pargs):
    model_name = args.model_name_or_path.split('/')[-1]
    log_folder = './logs/' + model_name
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file = f'{model_name}_{args.task}' + '_bs_' + str(args.bs) + '_maxlen_' + str(args.max_length) + '_lora_r_' + str(args.lora_r) + '_lora_alpha_' + str(args.lora_alpha) + '_lora_dropout_' + str(args.lora_dropout) \
        + '_module_lr_' + str(args.module_lr)+ '_head_lr_' + str(args.head_lr) \
        + '_beta_' + str(args.beta) + '_gemma_' + str(args.gemma) + '_weight_decay_' + str(args.weight_decay) + '_seed_' + str(args.seed) + '.txt'
    log_path = os.path.join(log_folder, log_file)
    # print(log_path)
    with open(log_path, mode = 'a+') as w:
        w.write(" ".join(["{}".format(t) for t in pargs]) + "\n")

# basic config 
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
task = args.task
if task == "stsb":
    num_labels = 1 # regression task
elif task == "mnli":
    num_labels = 3
else:
    num_labels = 2

# peft config
if args.peft == "kasa":
    peft_type = PeftType.LORA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # target_modules=["query", "value"], # specific the target modules
        bias="none",
        task_type="SEQ_CLS",
        inference_mode=False
    )
else:
    raise ValueError(f"peft {args.peft} is not supported.")


# tokenizer
if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side) if 'deberta' not in args.model_name_or_path else AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side, use_fast=False)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# load dataset and metrics
datasets = load_dataset("glue", task)
metric = load_metric("glue", task)

# tokenizing dataset
def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    if task == 'sst2' or task == 'cola':
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)
    elif task == 'qnli':
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=args.max_length)
    elif task == 'qqp':
        outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=args.max_length)
    elif task == 'mnli':
        outputs = tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=args.max_length)
    else:
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)
    return outputs

if task == 'sst2' or task == 'cola':
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
    )
elif task == 'qnli':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question", "sentence"],
    )
elif task == 'qqp':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question1", "question2"],
    )
elif task == 'mnli':
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "premise", "hypothesis"],
    ) 
else:
    tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
    )

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# wrapping tokenized datasets with DataLoader
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
# Following most of the PEFT papers (e.g. LoRA, VeRA, AdaLoRA ...), we report the best results on the validation set,
# because the labels of GLUE's test sets are not all available.
eval_dataloader = DataLoader(tokenized_datasets["validation" if task !="mnli" else "validation_matched"], shuffle=False, collate_fn=collate_fn, batch_size=args.bs)
if task == "mnli":
    eval_dataloader_mismatched = DataLoader(tokenized_datasets["validation_mismatched"], shuffle=False, collate_fn=collate_fn, batch_size=args.bs)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    import math
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if 'lora_diag' in name:
            all_param += int(math.sqrt(param.numel()))
        elif 'classifier' not in name:    
            all_param += param.numel()
        if param.requires_grad and 'classifier' not in name:
            if 'lora_diag' in name:
                print(name, int(math.sqrt(param.numel())))
                trainable_params += int(math.sqrt(param.numel()))
            else:
                print(name, param.numel())
                trainable_params += param.numel()
    print(f'trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {trainable_params/all_param}')
    return trainable_params, all_param

model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels, return_dict=True)
print(model)
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

# optimizer for parameters
head_param = list(map(id, model.classifier.parameters()))
others_param = filter(lambda p: id(p) not in head_param, model.parameters()) 
optimizer = AdamW([
    {"params": model.classifier.parameters(), "lr": args.head_lr},
    {"params": others_param, "lr": args.module_lr}
], weight_decay=args.weight_decay)

# learning rate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=args.warmup_ratio * (len(train_dataloader) * args.num_epochs),
    num_training_steps=(len(train_dataloader) * args.num_epochs),
)

# auxiliary loss
def loss_fn(model, beta=0.01, gamma=0.01, device='cuda'):
    l2_loss = 0.0
    l3_loss = 0.0
    block_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_diag' in name:
                block_num += 1
                diag_norm = torch.sum(param ** 2)
                l2_loss += diag_norm
            elif 'lora_A' in name or 'lora_B' in name:
                if 'lora_A' in name:
                    matmul_result = torch.matmul(param.T, param)
                else:  # 'lora_B' in name
                    matmul_result = torch.matmul(param, param.T)

                I = torch.eye(matmul_result.size(0), device=device)
                diff_I = matmul_result - I
                matrix_loss = torch.norm(diff_I, p='fro')
                l3_loss += matrix_loss

    auxi_loss = (beta * l2_loss + gamma * l3_loss) / block_num

    return auxi_loss 
 
acc_list = []
model.to(device)
for epoch in range(args.num_epochs):
    # model training
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        
        # auxiluary loss
        loss += loss_fn(model, args.beta, args.gemma, device)

        loss.backward() 
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # model evaluation
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if task != "stsb" else outputs.logits
        # print(outputs.logits)
        references = batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    # metrics calculation
    eval_metric = metric.compute() # returns a dictionary
    if task == "mnli":
        for step, batch in enumerate(tqdm(eval_dataloader_mismatched)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            # print(outputs.logits)
            references = batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric_mismatched = metric.compute() # returns a dictionary
    if task == "stsb":
        acc_list.append(eval_metric['pearson'])
        log(f"epoch {epoch}:", eval_metric, ', current_best_pearson:', max(acc_list),'train_loss:', loss)
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_pearson:\033[0m', max(acc_list),'train_loss:', loss.item())
    elif task == 'cola':
        acc_list.append(eval_metric['matthews_correlation'])
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_corr:\033[0m', max(acc_list),'train_loss:', loss.item())
        log(f"epoch {epoch}:", eval_metric, ', current_best_corr:', max(acc_list),'train_loss:', loss)
    elif task == 'mnli':
        acc_list.append((eval_metric['accuracy'] + eval_metric_mismatched['accuracy'])/2)
        print(f"epoch {epoch}:", {'matched': eval_metric, 'mismatched': eval_metric_mismatched}, '\033[32m, current_best_acc:\033[0m', max(acc_list),'train_loss:', loss.item())
        log(f"epoch {epoch}:", {'matched': eval_metric, 'mismatched': eval_metric_mismatched}, ', current_best_acc:', max(acc_list),'train_loss:', loss)
    else:
        acc_list.append(eval_metric['accuracy'])
        print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_acc:\033[0m', max(acc_list),'train_loss:', loss.item())
        log(f"epoch {epoch}:", eval_metric, ', current_best_acc:', max(acc_list),'train_loss:', loss)

