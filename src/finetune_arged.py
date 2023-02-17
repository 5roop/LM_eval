
import sys
import os
import argparse
import json
import random as python_random
import numpy as np
import torch
from simpletransformers.ner import NERModel
import pandas as pd


import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--train_file", required=True,
                    type=str,
                     help="Input file to learn from"
                     )
parser.add_argument("-t", "--test_file", type=str, required=True,
                    help="Files to test on."
                    )
parser.add_argument("-d", "--dev_file", type=str, required=True,
                    help="Files to test on."
                    )
parser.add_argument("-lt", "--lm_type", type=str, default="electra", required=True,
                    help="Simpletransformers LM type identifier")
parser.add_argument("-ln", "--lm_name", type=str, default="classla/bcms-bertic", required=True,
                    help="Simpletransformers LM name or path")
parser.add_argument("-s", "--seed", type=int, default=2222,
                    help="Random seed that we use")
parser.add_argument("--task", type=str, required=True,
                    help="Which task to evaluate on")
parser.add_argument("-o", "--output_file", default="eval.log", type=str,
                    help="What file to append the results to.")
parser.add_argument("-n", "--gpu_n", type=int, default=-1,
                    help="Which gpu device to use")
parser.add_argument("-e", "--num_train_epochs", type=int, default=10,
                    required=False,
                    help="Train epochs")
parser.add_argument("--lines", type=int, default=None,
                    help="How many lines of train data to use")
args = parser.parse_args()

np.random.seed(args.seed)
python_random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def get_train_args():
    '''Default training arguments, recommended to overwrite them with your own arguments
       They differ a bit from the simpletransformers default arguments, so beware!'''
    return {
    # "cache_dir": "cache_dir/",
    # "fp16": True,
    # "fp16_opt_level": "O1",
    # "max_seq_length": 512,
    # "train_batch_size": 8,
    # "gradient_accumulation_steps": 1,
    # "eval_batch_size": 8,
    "num_train_epochs": args.num_train_epochs,
    # "weight_decay": 0,
    # "learning_rate": 4e-5,
    # "adam_epsilon": 1e-8,
    # "warmup_ratio": 0.06,
    # "warmup_steps": 0,
    # "max_grad_norm": 1.0,
    # "logging_steps": 25,
    # "save_steps": 500000,
    "overwrite_output_dir": True,
    # "reprocess_input_data": False,
    # "evaluate_during_training": False,
    # "use_early_stopping": False,
    # "evaluate_during_training_verbose": True,
    # "evaluate_during_training_silent": False,
    # "evaluate_each_epoch": True,
    # "evaluate_during_training_steps": 500000,
    # "early_stopping_consider_epochs": True,
    # "early_stopping_delta": 0.0005,
    # "early_stopping_metric_minimize": False,
    # "early_stopping_patience": 2,
    # "early_stopping_metric": "f1_score",
    # "save_model_every_epoch": False,
    # "process_count": 8,
    # "no_cache": True,
    # "n_gpu": 1,
    }
    
# train_path = "/home/peterr/LanguageModels/finetune_data/hr_set-ud-train.conllu"
train_path = args.train_file
# dev_path = "/home/peterr/LanguageModels/finetune_data/hr_set-ud-dev.conllu"
dev_path = args.dev_file
# test_path = "/home/peterr/LanguageModels/finetune_data/hr_set-ud-test.conllu"
test_path = args.test_file

task = args.task

if (task == "upos") or (task == "xpos"):
    from utils import parse_conllu as parse
elif task == "ner":
    from utils import parse_ner as parse
    
train = parse(train_path, target=task)
if args.lines != None:
    train = train.head(args.lines)
dev = parse(dev_path, target=task)
test = parse(test_path, target=task)

# Extract all the labels that appear in the data:
labels = train.labels.unique().tolist() + test.labels.unique().tolist() + dev.labels.unique().tolist()
labels = list(set(labels))

from simpletransformers.ner import NERModel
# Create a NERModel
model_type = args.lm_type
model_name = args.lm_name
model = NERModel(model_type, model_name, labels=labels, 
                 cuda_device=args.gpu_n)
model.train_model(train_data=train, 
                #   eval_data=dev,
                  args = get_train_args())

results, model_outputs, predictions = model.eval_model(test)

from sklearn.metrics import f1_score, classification_report

# The model returns a list of lists, with the total count sometimes not adding up (model
# discards some instances, it seems. Happens to ~2 instances per test split, but then the evaluation crashes.)
# We unfold this list of lists, add it to original test data, and discard all of the sentences where
# there is a mismatch.
test["y_pred"] = ""
for i in test.sentence_id.unique():
    subset = test[test.sentence_id == i]
    if subset.shape[0] == len(predictions[i]):
        test.loc[test.sentence_id == i, "y_pred"] = predictions[i]
    else:
        continue
test = test[test.y_pred != ""]
y_true = test.labels.tolist()
y_pred = test.y_pred.tolist()

macrof1 = f1_score(y_true, y_pred, labels=labels, average='macro')
microf1 = f1_score(y_true, y_pred, labels=labels, average='micro')
clfreport = classification_report(y_true, y_pred, labels=labels)

print(clfreport)

from datetime import datetime
log = {"train": train_path,
       "dev": dev_path,
       "test": test_path,
       "model_type": model_type,
       "model_name": model_name,
       "microF1": microf1,
       "macroF1": macrof1,
       "task": task,
       "num_train_epochs": args.num_train_epochs,
       "num_lines": args.lines,
       "timestamp": datetime.now().__str__(),
       "train_args": get_train_args(),
       "seed": args.seed,
       "gpu_device": args.gpu_n,
       "labels": labels,
       "y_true": y_true,
       "y_pred": y_pred,
       "classification_report": "\n" + clfreport + "\n",
       }

with open(args.output_file, "a") as f:
    import json
    f.write(json.dumps(log) + "\n")