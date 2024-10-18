#!/usr/bin/env python
# coding: utf-8

# run with:
# deepspeed --num_gpus=8 pretrain_geneformer.py --deepspeed ds_config.json

import datetime
from time import time

# imports
import os
import sys

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"

GPU_NUMBER = [0, 1, 2, 3, 4, 5, 6, 7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])


import pickle
import random
import subprocess
import numpy as np
import pytz
from packaging import version
import argparse


import torch
import torch.distributed
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers.training_args import ParallelMode
from transformers.utils import is_tf_available, is_torch_available, logging, to_py_obj
from transformers.utils.generic import _is_tensorflow, _is_torch
from transformers.file_utils import is_datasets_available, is_sagemaker_dp_enabled
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, BertForNextSentencePrediction, BertForPreTraining, TrainingArguments, BertModel, AutoModel
from datasets import Dataset


from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union # 追加


from geneformer import GeneformerPretrainer


seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



def main(**kwargs) :

    # setting 
    mouse_geneformer_flag = kwargs.pop("mouse_geneformer_flag")
    use_pretrained = kwargs.pop("use_pretrained")
    change_dropout_rate = kwargs.pop("change_dropout_rate")


    # set local time/directories
    timezone = pytz.timezone("Asia/Tokyo")
    rootdir = "/path/to/root/directory"


    # set model parameters
    # model type
    model_type = "bert"                 # (default: bert)
    # Pre text task
    task = "MLM"                        # (choice pretext tasks MLM)
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
        # "pad_token_id": token_dictionary.get("<pad>"),
        # "vocab_size": len(token_dictionary),  # genes+special_tokens (<mask> and <pad> and so on... tokens)
    }
    
    # check the config if you use geneformer.
    if mouse_geneformer_flag == False :
        if task == "MLM" :
            pass
        else :
            print("geneformer doesn't train by {} task.".format(task))
            print("please choice MLM task.")
            sys.exit(1)
    else :
        pass

    # set training parameters
    # total number of examples in Genecorpus-30M after QC filtering:
    #　(default: 27_406_208) genecorpus-30M:27_406_208, mouse-genecorups-20M_data1-v2:21_332_982
    num_examples = 21_332_982
    # number gpus
    num_gpus = 8                # (default: 12)
    # batch size for training and eval
    geneformer_batch_size = 12   # (default: 12)
    # max learning rate
    max_lr = 1e-3               # (default: 1e-3)
    # learning schedule
    lr_schedule_fn = "cosine"   # (default: linear)
    # warmup steps
    warmup_steps = 10_000       # (default: 10_000)
    # number of epochs
    epochs = 10                 # (default: 3)
    # optimizer
    optimizer = "adamw_torch"         # (default: adamw)
    # weight_decay
    weight_decay = 0.001        # (default: 0.001)

    # training args
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
        "label_names": ["labels"] if task=="MLM" else ["next_sentence_label"] if task=="NSP" else ["labels", "next_sentence_label"] if task=="BERT" else ["labels"],
    }

    # check the config if you use geneformer.
    if mouse_geneformer_flag == False and task == "MLM" :
        if num_examples == 27_406_208 :
            pass
        else :
            print("you misstake genecorpus-30M cells.")
            print("please choice 27_406_208.")
            sys.exit(1)
    else :
        pass


    # Load datasets
    if mouse_geneformer_flag == True :
        dataset_version = "-n1"
        print("dataset_version: {}".format(dataset_version))
        if dataset_version == "-n1" :
            dataset_path = "/path/to/MLM-re_All_mouse_tokenize_dataset.dataset"
            dataset_length_path = "/path/to/MLM-re_All_mouse_tokenize_dataset_length.pkl"
            token_dictionary_path = "/path/to/MLM-re_token_dictionary_v1.pkl"
                
        else :
            print("select tasks in MLM right or select total cells (num_examples) right.")
            sys.exit(1)
    
    elif mouse_geneformer_flag == False :
        dataset_path = "/path/to/genecurpus_30M_2048.dataset"
        dataset_length_path = "/path/to/genecorpus_30M_2048_lengths.pkl"
        token_dictionary_path = "/path/to/token_dictionary_human_myocardial-covid19-ctchuman_mouse_cop1ko-easy-hard.pkl"
        
    else :
        print("select organism mouse or human")
        sys.exit(1)
    
    train_dataset=load_from_disk(dataset_path)

    # Load token directory
    with open(token_dictionary_path, "rb") as fp:
        token_dictionary = pickle.load(fp)
    
    
    # Add config
    config["pad_token_id"] = token_dictionary.get("<pad>")
    config["vocab_size"] = len(token_dictionary)


    # Using pretrain model or not using pretrain model
    if use_pretrained == False :
        use_or_not_use = "-NUse"
    else :
        use_or_not_use = "-Use"
    
    # Check the config if you use MLM task.
    if mouse_geneformer_flag == False and task == "MLM" and num_examples == 27_406_208 :
        if use_or_not_use == "-NUse":
            pass
        else :
            print("geneformer doesn't use pretrained model.")
            print("please choice 'False' in the use_pretrained.")
            sys.exit(1)
    else :
        pass
    
    
    # Generate saving path of model and logdata
    current_date = datetime.datetime.now(tz=timezone)
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
    if mouse_geneformer_flag == True :
        run_name = f"{datestamp}_mouse-geneformer_PM{use_or_not_use}_20M_DV{dataset_version}_T{task}_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_DR{hidden_dropout_prob}_ACT{activ_fn}_O{optimizer}_DS{num_gpus}"
    else :
        run_name = f"{datestamp}_geneformer_PM{use_or_not_use}_30M_DV{dataset_version}_T{task}_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_DR{hidden_dropout_prob}_ACT{activ_fn}_O{optimizer}_DS{num_gpus}"
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


    # Add training args
    training_args["output_dir"] = training_output_dir
    training_args["logging_dir"] = logging_dir

    # Generate training_args
    training_args = TrainingArguments(**training_args)
    
    
    print("Generate the Bert model!")
    # Generate BertConfig
    config = BertConfig(**config)

    # load model
    if task == "MLM" :
        model = BertForMaskedLM(config) # Masked Language Modeling (MLM)
    else :
        pass
    

    # model mode is train        
    model = model.train()

    pretrain_model_name = "mouse-Geneformer" if mouse_geneformer_flag==True else "Geneformer"
    print("Starting {} training by {} task.".format(pretrain_model_name, task))

    # Define trainer
    trainer = GeneformerPretrainer(
        model=model,
        args=training_args,
        # pretraining corpus (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)
        train_dataset=train_dataset,
        # file of lengths of each example cell (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/genecorpus_30M_2048_lengths.pkl)
        example_lengths_file=dataset_length_path,
        token_dictionary=token_dictionary,
        pretext_task = task,
    )

    # Start training
    start_time = time()
    if epochs == 0 :
        pass
    else :
        trainer.train()
    print(f"Finished training Geneformer. Total tim: {time() - start_time}")

    # save model
    trainer.save_model(model_output_dir)
    print(f"Saved the model: {model_output_dir}")

    # evaluate
    #trainer.evaluate()
    #print("Finished evaluating Geneformer.")


    return 0



if __name__ =="__main__" :

    logger = logging.get_logger(__name__)
    EncodedInput = List[int]
    VERY_LARGE_INTEGER = int(
        1e30
    )  # This is used to set the max input length for a model with infinite size input
    LARGE_INTEGER = int(
        1e20
    )  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

    if is_sagemaker_dp_enabled():
        import smdistributed.dataparallel.torch.distributed as dist
    else:
        import torch.distributed as dist

    _is_torch_generator_available = False
    if version.parse(torch.__version__) >= version.parse("1.6"):
        _is_torch_generator_available = True

    
    # Flag of geneformer or mouse-Geneformer 
    mouse_flag = True

    # Falg of using pretrained model or not using pretrained mode 
    use_pretrained = False

    # change dropout rate
    change_dropout_rate = False
    
    main(mouse_geneformer_flag=mouse_flag, use_pretrained = use_pretrained, change_dropout_rate = change_dropout_rate)


