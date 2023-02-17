#!/bin/bash

model_type="--lm_type roberta"
output="--output_file eval_reports.jsonl"
models=$(ls models/ | grep checkpoint)
# lines="--lines 10000"
lines=""
epochs="--num_train_epochs 10"

hr_conlls="--train_file finetune_data/hr_set-ud-train.conllu --dev_file finetune_data/hr_set-ud-dev.conllu --test_file finetune_data/hr_set-ud-test.conllu"

for task in xpos upos
do
    for model in $models
    do
        python petersrc/finetune_arged.py $hr_conlls $model_type --lm_name "models/$model" --seed 2222 $output --task $task --gpu_n 5 $lines $epochs
    done
done

sr_conlls="--train_file finetune_data/sr_set-ud-train.conllu --dev_file finetune_data/sr_set-ud-dev.conllu --test_file finetune_data/sr_set-ud-test.conllu"

for task in xpos upos
do
    for model in $models
    do
        python petersrc/finetune_arged.py $sr_conlls $model_type --lm_name "models/$model" --seed 2222 $output --task $task --gpu_n 6 $lines $epochs
    done
done

task="ner"
epochs="--num_train_epochs 10"
hr_ner="--train_file finetune_data/hr500k-train.ner --dev_file finetune_data/hr500k-dev.ner --test_file finetune_data/hr500k-test.ner"

for model in $models
do
    python petersrc/finetune_arged.py $hr_ner $model_type --lm_name "models/$model" --seed 2222 $output --task $task --gpu_n 6 $lines $epochs
done

sr_ner="--train_file finetune_data/set.sr.plus-train.ner --dev_file finetune_data/set.sr.plus-dev.ner --test_file finetune_data/set.sr.plus-test.ner"

for model in $models
do
    python petersrc/finetune_arged.py $sr_ner $model_type --lm_name "models/$model" --seed 2222 $output --task $task --gpu_n 6 $lines $epochs
done
