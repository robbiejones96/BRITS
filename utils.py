import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import pandas as pd

import yaml
import json
import os

def read_json(json_file):
    json_data = None
    print("Loading json data from {}".format(os.path.abspath(json_file)))
    with open(json_file, 'r') as input_file:
        json_data = json.load(input_file)
    return json_data

def save_json(json_data, data_folder, json_file_name):
    output_file_path = os.path.join(data_folder, json_file_name + ".json")
    print("Saving JSON to {}".format(os.path.abspath(output_file_path)))
    with open(output_file_path, 'w') as output_file:
        json.dump(json_data, output_file)
    return output_file_path

def read_yaml(yaml_file):
    yaml_data = None
    print("Loading yaml data from {}".format(os.path.abspath(yaml_file)))
    with open(yaml_file, 'r') as input_file:
        yaml_data = yaml.load(input_file, Loader=yaml.Loader)
    return yaml_data

def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = [to_var(x) for x in var]
        return var

def stop_gradient(x):
    if isinstance(x, float):
        return x
    if isinstance(x, tuple):
        return tuple(map(lambda y: Variable(y.data), x))
    return Variable(x.data)

def zero_var(sz):
    x = Variable(torch.zeros(sz))
    if torch.cuda.is_available():
        x = x.cuda()
    return x
