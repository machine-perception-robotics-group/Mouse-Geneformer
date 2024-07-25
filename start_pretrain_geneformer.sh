#!/bin/bash
export PATH=$PATH:~/.local/bin
deepspeed --num_gpus=8 pretrain_geneformer_w_deepspeed_one_dataset_ver.py --deepspeed ds_config.json