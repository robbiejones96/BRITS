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
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --sample-size=<int>                     sample size [default: 5]
    --dropout=<float>                       dropout [default: 0.3]
    --log-file=<log-file>                   name of log file to log all output
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

def plot_losses(losses, experiment_name, results_folder):
    loss_graph_save_path = os.path.join(results_folder, "loss_graph.png")
    train_losses, val_losses = losses
    plt.plot(np.arange(len(train_losses)), train_losses, label="Train loss")
    plt.plot(np.arange(len(val_losses)), val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("{}: Training and validation losses".format(experiment_name))
    plt.legend(loc = "upper right")
    log("Saving loss graph to {}".format(loss_graph_save_path))
    plt.savefig(loss_graph_save_path)

def log(string):
    """
        A convenience function to print to standard output as well as write to
        a log file. Note that LOG_FILE is a global variable set in main and
        is specified by a command line argument. See docopt string at top
        of this file.

        param string (str): string to print and log to file
    """
    if LOG_FILE is not None:
        LOG_FILE.write(string + '\n')
    print(string)

def run_on_data(model, data_iter, optimizer):
    loss = 0.0
    for idx, data in enumerate(data_iter):
        ret = model.run_on_batch(data, optimizer)
        loss += ret['loss'].item()
    return loss / len(data_iter)

def run_training(model, optimizer, train_iter, val_iter, max_epoch):
    train_losses = []
    val_losses = []
    for epoch in range(5):
        log("Epoch {}".format(epoch))
        model.train()
        train_loss = run_on_data(model, train_iter, optimizer)
        model.eval()
        val_loss = run_on_data(model, val_iter, None)
        log("Train loss: {}, Val loss: {}".format(train_loss, val_loss))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

                
    return train_losses, val_losses

def set_random_seed(seed):
    if seed:
        log("Running training with random seed {}".format(seed))
        torch.manual_seed(seed)
    else:
        log("WARNING: The random number generator has not been seeded.")
        log("You are encouraged to run with a random seed for reproducibility!")

def set_cuda(model, no_cuda):
    if not no_cuda and torch.cuda.is_available():
            log("CUDA is available, using GPU...")
            log("Pass --no-cuda to main.py if you don't want GPU")
            return model.cuda()
    else:
        log("Using CPU...")
        return model

def load_optimizer(model, optimizer_name, lr):
    return getattr(optim, optimizer_name)(model.parameters(), lr=lr)

def save_model(model, results_folder):
    model_save_path = os.path.join(results_folder, "trained_model.pt")
    log("Saving model to {}".format(model_save_path))
    torch.save(model.state_dict(), model_save_path)

def train(args, yaml_data):
    

    set_random_seed(yaml_data["seed"])
    train_iter, model = load_data_and_model(yaml_data, "train_data")
    model = set_cuda(model, args["--no-cuda"])

    batch_size = yaml_data["batch_size"]
    val_iter, _ = data_loader.get_loader(yaml_data["val_data"], batch_size)

    optimizer = load_optimizer(model, yaml_data["optimizer"], yaml_data["lr"])

    max_epoch = yaml_data["max_epoch"] 
    losses = run_training(model, optimizer, train_iter, val_iter, max_epoch)
    save_model(model, yaml_data["results_folder"])
    plot_losses(losses, yaml_data["experiment_name"], yaml_data["results_folder"])

    if args["--evaluate"]:
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

    log("MAE: {}".format(np.abs(evals - imputations).mean()))
    log("MRE: {}".format(np.abs(evals - imputations).sum() / np.abs(evals).sum()))
    log("NRMSE: {}".format(np.sqrt(np.power(evals - imputations, 2).mean()) / (evals.max() - evals.min())))

def load_data_and_model(yaml_data, data_type):
    data_path = yaml_data[data_type]
    batch_size = yaml_data["batch_size"]
    model_name = yaml_data["model"]
    hidden_size = yaml_data["hidden_size"]
    data_iter, input_dim = data_loader.get_loader(data_path, batch_size)
    model = globals()[model_name](input_dim = input_dim, hidden_size = hidden_size)
    return data_iter, model

def prep_eval(yaml_data, no_cuda):
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

def set_log_file(log_file, results_folder):
    """
        Opens a writeable file object to log output and sets as a global
        variable. 

        :param log_file (str): name of log file to write to, can be None
            * if None, no log file is created
        :param results_folder (str): folder to save log file to
    """
    global LOG_FILE
    LOG_FILE = None
    if log_file is not None:
        log_file_path = os.path.join(results_folder, log_file + ".txt")
        LOG_FILE = open(log_file_path, 'w')
        log("Created log file at {}".format(log_file_path))

def main():
    args = docopt(__doc__)
    yaml_data = read_yaml(args["--yaml-file"])
    set_log_file(args["--log-file"], yaml_data["results_folder"])

    if args["train"]:
        train(args, yaml_data)
    elif args["evaluate"]:
        yaml_data = read_yaml(yaml_file)
        data_iter, model = prep_eval(yaml_data, args["--no-cuda"])
        evaluate(model, data_iter, yaml_data["results_folder"])

    if LOG_FILE is not None:
        LOG_FILE.close()
        

if __name__ == '__main__':
    main()
