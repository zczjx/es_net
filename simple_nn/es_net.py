#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
from active_func import *

class es_net(object):

    def __init__(self, layers, active_func=sigmoid):
        self.layers = layers
        self.weight = list()
        self.bias = list()
        self.active_func = active_func;
        
    def setup_nn_param(self, layer_idx, weight_arr, bias_arr):
        self.weight.append(weight_arr)
        self.bias.append(bias_arr)

    def forward_inference(self, input_x):
        n = 0
        z = input_x
        while n < self.layers:
            a = np.dot(z, self.weight[n]) + self.bias[n]
            z = self.active_func(a)
            n = n + 1
        y = self.identity_fun(z)
        return y
    
    def identity_fun(self, x):
        return x