#!/bin/bash
export PATH=$PATH:~/.local/bin
deepspeed --num_gpus=8 pretrain_geneformer.py --deepspeed ds_config.json