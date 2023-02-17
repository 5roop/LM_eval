#!/bin/bash

# Proposed file structure:
# LM_eval:
# -finetune_data/... 
# -models/...
# -src/...
# this_script.sh


# Our models are roberta, Bertic is electra:
model_type="--lm_type roberta" 

# Model path (or name, if published):
model_path="--lm_name models/checkpoint-2500"

# Where would you like the report to be appended?
output="--output_file eval_reports.jsonl" 


# Use the commented line to clip the train data at some number of lines. Use the uncommented line for ful train data.
# lines="--lines 10000"
lines=""

# Good luck with this one. If it'll not train for set number of epochs,
# you can try setting 9 more.
epochs="--num_train_epochs 19"

# Set the files
hr_conlls="--train_file finetune_data/hr_set-ud-train.conllu --dev_file finetune_data/hr_set-ud-dev.conllu --test_file finetune_data/hr_set-ud-test.conllu"
sr_conlls="--train_file finetune_data/sr_set-ud-train.conllu --dev_file finetune_data/sr_set-ud-dev.conllu --test_file finetune_data/sr_set-ud-test.conllu"

# Set the task (can be upos, xpos, ner)
task="upos"

# The rest of the switches, explained briefly:
# --seed 2222: seed for reproducibility
# --gpu_n 6: which GPU to use

# Run the command on HR data:
python src/finetune_arged.py $hr_conlls $model_type $model_path --seed 2222 $output --task $task --gpu_n 6 $lines $epochs

# Repeat the command for SR data:
python src/finetune_arged.py $sr_conlls $model_type $model_path --seed 2222 $output --task $task --gpu_n 6 $lines $epochs
