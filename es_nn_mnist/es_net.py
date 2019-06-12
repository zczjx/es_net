#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
from active_func import *
from common_func import *

class es_net(object):

    def __init__(self, layers, active_func=sigmoid, out_func=identity_equal):
        self.layers = layers
        self.weight = []
        self.bias = []
        self.active_func = active_func;
        self.output_func = out_func
        
    def setup_nn_param(self, layer_idx, weight_arr, bias_arr):
        self.weight.append(weight_arr)
        self.bias.append(bias_arr)

    def forward_inference(self, input_x):
        n = 0
        z = input_x

        for n in range(0, self.layers, 1):
            a = np.dot(z, self.weight[n]) + self.bias[n]
            z = self.active_func(a)
            
        y = self.output_func(z)
        return y
