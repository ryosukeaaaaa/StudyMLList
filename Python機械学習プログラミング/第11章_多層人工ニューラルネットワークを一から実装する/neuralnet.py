# Python Machine Learning by Sebastian Raschka, Packt Publishing Ltd. 2015
# Code Repository: https://github.com/rasbt/python-machine-learning-book
# Code License: MIT License

import numpy as np
from scipy.special import expit
import sys

def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1.0
    return ary

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes

        # 隠れ層
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(loc=0.0, scale=0.1,
                                   size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # 出力層
        self.weight_out = rng.normal(loc=0.0, scale=0.1,
                                     size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        # 隠れ層：
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # 出力層：
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        #########################
        # one-hot エンコーディング
        #########################
        y_onehot = int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/doutWeights =
        #         dLoss/dOut * dOutAct/dOutNet * dOutNet/doutWeight
        # => const= DeltaOut = dLoss/dOut * dOutAct/dOutNet（再利用に役立つ）

        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out) # シグモイド関数の微分

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # 出力層の重みについての勾配
        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h

        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__d_w_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__d_b_out = np.sum(delta_out, axis=0)

        # Part 2: dLoss/dHiddenWeights =
        #         DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet *
        #         dHiddenNet/dWeight

        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out

        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. - a_h)     # シグモイド関数の微分

        # [n_examples, n_features]
        d_z_h__d_w_h = x

        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h)