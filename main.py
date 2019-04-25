#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    main.py train --train-src=<train-src> [options]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --model=<name>                          model to use (rits, brits, rits_i, brits_i) [default: brits]
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 1000]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
"""

from docopt import docopt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

import time
import utils
import models.rits_i
import argparse
import data_loader
import pandas as pd
import ujson as json

from sklearn import metrics
from ipdb import set_trace


def train(model_name, args):
    data_iter, dim = data_loader.get_loader(args["--train-src"], batch_size = int(args["--batch-size"]))
    model = getattr(models, model_name).Model(dim)
    optimizer = optim.Adam(model.parameters(), float(args["--lr"]))
    for epoch in range(int(args["--max-epoch"])):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            ret = model.run_on_batch(data, optimizer)

            run_loss += ret['loss'].item()

            print("\r Progress epoch {}, {:.2f}%, average loss {}".format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))

        if run_loss < 20:
            print(ret["imputations"])
            break

    evaluate(model, data_iter)

def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['labels'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)

    print('AUC {}'.format(metrics.roc_auc_score(labels, preds)))

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    print('MAE', np.abs(evals - imputations).mean())
    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())

def run():
    args = docopt(__doc__)
    

    if args["--cuda"]:
        model = model.cuda()

    train(args["--model"], args)

if __name__ == '__main__':
    run()
