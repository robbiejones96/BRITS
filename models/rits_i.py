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

from ipdb import set_trace
from sklearn import metrics

SEQ_LEN = 49
RNN_HID_SIZE = 64

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


class TemporalDecay(nn.Module):
    def __init__(self, input_size):
        super(TemporalDecay, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(RNN_HID_SIZE, input_size))
        self.b = Parameter(torch.Tensor(RNN_HID_SIZE))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model(nn.Module):
    def __init__(self, input_dim = 35):
        super(Model, self).__init__()
        self.build(input_dim)

    def build(self, input_dim):
        # we concatenate input with mask vector for input_dim * 2 size
        self.rnn_cell = nn.LSTMCell(input_dim * 2, RNN_HID_SIZE)

        self.regression = nn.Linear(RNN_HID_SIZE, input_dim)
        self.temp_decay = TemporalDecay(input_size = input_dim)

        self.out = nn.Linear(RNN_HID_SIZE, 1)

    def forward(self, data, direction):
        """
        Passes batched data through the RITS algorithm (4.1.1 in the paper)

        :param data: (dict) stores time series data for forward and backward directions
        :param direction: (str) either "forward" or "backward"
        :returns: (dict) stores
            * loss: (float) MAE + classification loss (if applicable)
            * predictions: (Tensor) predicted labels
            * imputations: 
        """
        x_t = data[direction]["x_t"]
        masks = data[direction]["masks"]
        deltas = data[direction]["deltas"]

        evals = data[direction]["evals"]
        eval_masks = data[direction]["eval_masks"]

        is_train = data["is_train"].view(-1, 1)

        h = torch.zeros((x_t.size()[0], RNN_HID_SIZE))
        c = torch.zeros((x_t.size()[0], RNN_HID_SIZE))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []
        seq_len = x_t.size()[1]
        for t in range(seq_len):
            x = x_t[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            
            x_hat = self.regression(h)

            x_c =  m * x +  (1 - m) * x_hat

            x_loss += torch.sum(torch.abs(x - x_hat) * m) / (torch.sum(m) + 1e-5)

            inputs = torch.cat([x_c, m], dim = 1)

            gamma = self.temp_decay(d)
            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(x_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        # y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)

        # # only use training labels
        # y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss': x_loss / seq_len + 0.1 *y_loss, 'predictions': y_h,\
                'imputations': imputations, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer):
        ret = self(data, direction = "forward")

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
