
from sklearn.metrics import f1_score, classification_report
from utils import parse_conllu as parse
import warnings
import pandas as pd
from simpletransformers.ner import NERModel
import torch
import numpy as np
import random as python_random
import json
import argparse
import os
from pathlib import Path
import sys


def get_train_args():
    '''Default training arguments, recommended to overwrite them with your own arguments
       They differ a bit from the simpletransformers default arguments, so beware!'''
    return {
        "num_train_epochs": 5,
        "overwrite_output_dir": True,
        "process_count": 1,
    }


warnings.filterwarnings("ignore")

np.random.seed(2222)
python_random.seed(2222)
torch.manual_seed(2222)
torch.cuda.manual_seed(2222)
torch.cuda.manual_seed_all(2222)

# Croatian:
train_path = "finetune_data/hr_set-ud-train.conllu"
dev_path = "finetune_data/hr_set-ud-dev.conllu"
test_path = "finetune_data/hr_set-ud-test.conllu"

task = "upos"
train = parse(train_path, target=task)
dev = parse(dev_path, target=task)
test = parse(test_path, target=task)

# Extract all the labels that appear in the data:
labels = train.labels.unique().tolist() + test.labels.unique().tolist() + \
    dev.labels.unique().tolist()
labels = list(set(labels))


# Create a NERModel
model_type = "roberta"
print(f"Current working dir: {Path('.').absolute()}")
model_name = input("Input path to model: ")
model = NERModel(model_type, model_name, labels=labels,
                 cuda_device=-1)
model.train_model(train_data=train,
                  args=get_train_args())

results, model_outputs, predictions = model.eval_model(test)


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
print(f"{'Croatian':*^53}")
print(clfreport)


# Serbian:
train_path = "finetune_data/sr_set-ud-train.conllu"
dev_path = "finetune_data/sr_set-ud-dev.conllu"
test_path = "finetune_data/sr_set-ud-test.conllu"

train = parse(train_path, target=task)
dev = parse(dev_path, target=task)
test = parse(test_path, target=task)

# Extract all the labels that appear in the data:
labels = train.labels.unique().tolist() + test.labels.unique().tolist() + \
    dev.labels.unique().tolist()
labels = list(set(labels))
model = NERModel(model_type, model_name, labels=labels,
                 cuda_device=-1)
model.train_model(train_data=train,
                  args=get_train_args())
results, model_outputs, predictions = model.eval_model(test)
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
print(f"{'Serbian':*^53}")
print(clfreport)
