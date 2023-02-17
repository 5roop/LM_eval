#!/bin/bash

# Usage:
# python src/finetune_arged.py --help
# usage: finetune_arged.py [-h] -i TRAIN_FILE -t TEST_FILE -d DEV_FILE -lt LM_TYPE -ln LM_NAME [-s SEED] --task TASK
#                          [-o OUTPUT_FILE] [-n GPU_N] [-e NUM_TRAIN_EPOCHS] [--lines LINES]

# optional arguments:
#   -h, --help            show this help message and exit
#   -i TRAIN_FILE, --train_file TRAIN_FILE
#                         Input file to learn from
#   -t TEST_FILE, --test_file TEST_FILE
#                         Files to test on.
#   -d DEV_FILE, --dev_file DEV_FILE
#                         Files to test on.
#   -lt LM_TYPE, --lm_type LM_TYPE
#                         Simpletransformers LM type identifier
#   -ln LM_NAME, --lm_name LM_NAME
#                         Simpletransformers LM name or path
#   -s SEED, --seed SEED  Random seed that we use
#   --task TASK           Which task to evaluate on
#   -o OUTPUT_FILE, --output_file OUTPUT_FILE
#                         What file to append the results to.
#   -n GPU_N, --gpu_n GPU_N
#                         Which gpu device to use
#   -e NUM_TRAIN_EPOCHS, --num_train_epochs NUM_TRAIN_EPOCHS
#                         Train epochs
#   --lines LINES         How many lines of train data to use
python src/finetune_arged.py \
    --train_file finetune_data/hr500k-train.ner \
    --dev_file finetune_data/hr500k-dev.ner \
    --test_file finetune_data/hr500k-test.ner \
    --task ner \
    --lm_type roberta \
    --lm_name models/checkpoint-2500
    --num_train_epochs 1

