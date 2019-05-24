import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

from sklearn import metrics
from pdb import set_trace

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        # constrain weight matrix to have zeros along diagonal
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(Model, self).__init__()
        self.build(input_dim, hidden_size)

    def build(self, input_dim, hidden_size):
        self.hidden_size = hidden_size
        self.rnn_cell = nn.LSTMCell(input_dim * 2, hidden_size)

        self.temp_decay_h = TemporalDecay(input_size = input_dim, output_size = hidden_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = input_dim, output_size = input_dim, diag = True)

        self.hist_reg = nn.Linear(hidden_size, input_dim)
        self.feat_reg = FeatureRegression(input_dim)

        self.weight_combine = nn.Linear(input_dim * 2, input_dim)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, data, direction):
        """
        Passes batched data through the RITS algorithm (4.1.1 in the paper)

        :param data: (dict) storing time series data for forward and backward directions
        :param direction: (str) either "forward" or "backward"
        :returns: (dict) storing
            * loss: (float) MAE + classification loss (if applicable)
            * predictions: (Tensor) storing predicted labels
            * imputations: 
        """

        x_t = data[direction]["x_t"]
        masks = data[direction]["masks"]
        deltas = data[direction]["deltas"]

        if next(self.parameters()).is_cuda:
            x_t = x_t.cuda()
            masks = masks.cuda()
            deltas = deltas.cuda()
        
        evals = data[direction]["evals"]
        eval_masks = data[direction]["eval_masks"]

        h = torch.zeros((x_t.size()[0], self.hidden_size))
        c = torch.zeros((x_t.size()[0], self.hidden_size))

        if next(self.parameters()).is_cuda:
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0

        imputations = []

        seq_len = x_t.size()[1]
        for t in range(seq_len):
            x = x_t[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))
            

        imputations = torch.cat(imputations, dim = 1)

        return {'loss': x_loss / seq_len, 'imputations': imputations, \
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer):
        ret = self(data, direction = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
