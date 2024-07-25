#!/usr/bin/env python
# coding: utf-8

# run with:
# deepspeed --num_gpus=12 --num_nodes=3 pretrain_geneformer_w_deepspeed.py --deepspeed ds_config.json

import datetime

# imports
import os
import sys

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

import pickle
import random
import subprocess

import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, BertForNextSentencePrediction, BertForPreTraining, TrainingArguments

from geneformer import GeneformerPretrainer
from time import time
from torch.utils.tensorboard import SummaryWriter

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# set local time/directories
timezone = pytz.timezone("your/area")
rootdir = "/path/to/this/program/"


# set model parameters
# model type
model_type = "bert"                 # (default: bert)
# max input size
max_input_size = 2**11              # (default: 2**11 = 2048) 
# number of layers
num_layers = 6                      # (default: 6)
# number of attention heads
num_attn_heads = 4                  # (default: 4)
# number of embedding dimensions
num_embed_dim = 256                 # (default: 256)
# intermediate size
intermed_size = num_embed_dim * 2   # (default: num_embed_dim * 2)
# activation function
activ_fn = "silu"                   # (default: relu)
# initializer range, layer norm, dropout
initializer_range = 0.02            # (default: 0.02)
layer_norm_eps = 1e-12              # (default: 1e-12)
attention_probs_dropout_prob = 0.02 # (default: 0.02)
hidden_dropout_prob = 0.02          # (default: 0.02)


# set training parameters
# total number of examples in Genecorpus-30M after QC filtering:
num_examples = 21_332_982   #　(default: 21_332_982) 
# number gpus
num_gpus = 8                # (default: 8)
# batch size for training and eval
geneformer_batch_size = 12   # (default: 12)
# max learning rate
max_lr = 1e-3               # (default: 1e-3)
# learning schedule
lr_schedule_fn = "cosine"   # (default: cosine)
# warmup steps
warmup_steps = 10_000       # (default: 10_000)
# number of epochs
epochs = 10                 # (default: 10)
# optimizer
optimizer = "adamw"         # (default: adamw)
# weight_decay
weight_decay = 0.001        # (default: 0.001)


# path to saved mouse-Genecorpus-20M and it length file
dataset_path = "/path/to/saved/mouse-Genecorups-20M/MLM-re_All_mouse_tokenize_dataset.dataset"
dataset_length_path = "/path/to/saved/mouse-Genecorups-20M/lenght/MLM-re_All_mouse_tokenize_dataset_length.pkl"
print(f"dataset_path: {dataset_path}")
print(f"dataset_length_path: {dataset_length_path}")

# load gene_ensembl_id:token dictionary (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/token_dictionary.pkl)
with open("/path/to/token/dictionary/MLM-re_token_dictionary_v1.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)

# output directories
current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = f"{datestamp}_mouse-geneformer_20M_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_ACT{activ_fn}_O{optimizer}_GPUS{num_gpus}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"   
model_output_dir = os.path.join(training_output_dir, "models/")

# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")

# make training and model output directories
subprocess.call(f"mkdir {training_output_dir}", shell=True)
subprocess.call(f"mkdir {model_output_dir}", shell=True)


# model configuration
config = {
    "hidden_size": num_embed_dim,
    "num_hidden_layers": num_layers,
    "initializer_range": initializer_range,
    "layer_norm_eps": layer_norm_eps,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "hidden_dropout_prob": hidden_dropout_prob,
    "intermediate_size": intermed_size,
    "hidden_act": activ_fn,
    "max_position_embeddings": max_input_size,
    "model_type": model_type,
    "num_attention_heads": num_attn_heads,
    "pad_token_id": token_dictionary.get("<pad>"),
    "vocab_size": len(token_dictionary),  # genes+special_tokens (<mask> and <pad> and so on... tokens)
}

config = BertConfig(**config)

# choice Bert models ( https://huggingface.co/docs/transformers/ja/model_doc/bert#transformers.BertForPreTraining )
model = BertForMaskedLM(config) # Masked Language Modeling (MLM)    
model = model.train()

# define the training arguments
training_args = {
    "learning_rate": max_lr,
    "do_train": True,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay,
    "per_device_train_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "save_strategy": "steps",
    "save_steps": np.floor(num_examples / geneformer_batch_size / 8),  # 8 saves per epoch
    "logging_steps": 3,
    "output_dir": training_output_dir,
    "logging_dir": logging_dir,
    "label_names": ["labels"],
}
training_args = TrainingArguments(**training_args)

print("Starting mouse-Geneformer training by MLM task.")

# define the trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    # pretraining corpus (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)
    train_dataset=load_from_disk(dataset_path),
    # file of lengths of each example cell (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/genecorpus_30M_2048_lengths.pkl)
    example_lengths_file=dataset_length_path,
    token_dictionary=token_dictionary,
)

# train
start_time = time()
if epochs == 0 :
    # not pretraining
    pass
else :
    trainer.train()
print(f"Finished training Geneformer. Total tim: {time() - start_time}")


# save model
trainer.save_model(model_output_dir)
print(f"Saved the model: {model_output_dir}")
