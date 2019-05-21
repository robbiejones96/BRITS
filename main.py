#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    main.py train --yaml-file=<yaml-file> [--evaluate] [options]
    main.py evaluate --yaml-file=<yaml-file> [options]
    main.py test --model-path=<model-path> [--results-folder=<results-folder>]

Options:
    -h --help                               show this screen.
    --no-cuda                               don't use GPU (by default will try to use GPU)
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

import os
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

from models.brits import Model as BRITS
from models.rits import Model as RITS

import argparse
import data_loader

from sklearn import metrics
import matplotlib.pyplot as plt

from utils import read_json, save_json, read_yaml
from pdb import set_trace

def train(yaml_file, no_cuda, evaluate_after):
    yaml_data = read_yaml(yaml_file)
    batch_size = yaml_data["batch_size"]
    optimizer_name = yaml_data["optimizer"]
    lr = yaml_data["learning_rate"]
    max_epoch = yaml_data["max_epoch"]
    seed = yaml_data["seed"]
    val_data = yaml_data["val_data"]
            

    if seed:
        print("Running training with random seed {}".format(seed))
        torch.manual_seed(seed)
    else:
        print("WARNING: The random number generator has not been seeded.")
        print("You are encouraged to run again with a random seed for reproducibility!")

    data_iter, model = load_data_and_model(yaml_data, "train_data")
    if not no_cuda and torch.cuda.is_available():
            print("CUDA is available, using GPU...")
            print("Pass --no-cuda as an argument to main.py if you don't want GPU")
            model = model.cuda()
    else:
        print("Using CPU...")

    val_iter, _ = data_loader.get_loader(val_data, batch_size)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    for epoch in range(max_epoch):
        model.train()

        train_loss = 0.0

        for idx, data in enumerate(data_iter):
            ret = model.run_on_batch(data, optimizer)

            train_loss += ret['loss'].item()

            print("\r Progress epoch {}, {:.2f}%, average loss {}".format(epoch, (idx + 1) * 100.0 / len(data_iter), train_loss / (idx + 1.0)))

        if train_loss < 20:
            print(ret["imputations"])
            break

        train_losses.append(train_loss / len(data_iter))
        val_loss = 0.0
        for idx, data in enumerate(val_iter):
            ret = model.run_on_batch(data, None)

            val_loss += ret['loss'].item()
        val_losses.append(val_loss / len(val_iter))


    results_folder = yaml_data["results_folder"]
    model_save_path = os.path.join(results_folder, "trained_model.pt")
    print("Saving model to {}".format(model_save_path))
    torch.save(model.state_dict(), model_save_path)
    loss_graph_save_path = os.path.join(results_folder, "loss_graph.png")
    plt.plot(np.arange(len(train_losses)), train_losses, label="Train loss")
    plt.plot(np.arange(len(val_losses)), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Training and validation losses")
    plt.legend(loc = "upper right")
    print("Saving loss graph to {}".format(loss_graph_save_path))
    plt.savefig(loss_graph_save_path)

    if evaluate_after:
        if "val_data" not in yaml_data:
            print("Validation data not found! Check the yaml file you provided.")
        else:
            val_data = yaml_data["val_data"]
            data_iter, _ = data_loader.get_loader(val_data, batch_size)
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

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

    
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    print('MAE', np.abs(evals - imputations).mean())
    print('MRE', np.abs(evals - imputations).sum() / np.abs(evals).sum())
    print("NRMSE", np.sqrt(np.power(evals - imputations, 2).mean()) / (evals.max() - evals.min()))

def load_data_and_model(yaml_data, data_type):
    data_path = yaml_data[data_type]
    batch_size = yaml_data["batch_size"]
    model_name = yaml_data["model"]
    hidden_size = yaml_data["hidden_size"]
    data_iter, input_dim = data_loader.get_loader(data_path, batch_size)
    model = globals()[model_name](input_dim = input_dim, hidden_size = hidden_size)
    return data_iter, model

def prep_eval(yaml_file, no_cuda):
    yaml_data = read_yaml(yaml_file)
    data_iter, model = load_data_and_model(yaml_data, "val_data")
    results_folder = yaml_data["results_folder"]
    model_save_path = os.path.join(results_folder, "trained_model.pt")
    if not no_cuda and torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    map_location = device if device_name == "cpu" else None
    model.load_state_dict(torch.load(model_save_path, map_location = map_location))
    if device_name == "gpu": 
        model.to(device)
    return data_iter, model
    

def run():
    args = docopt(__doc__)

    if args["train"]:
        train(args["--yaml-file"], args["--no-cuda"], args["--evaluate"])
    elif args["evaluate"]:
        data_iter, model = prep_eval(args["--yaml-file"], args["--no-cuda"])
        evaluate(model, data_iter)
        

if __name__ == '__main__':
    run()
