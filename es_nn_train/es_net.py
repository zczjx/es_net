#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
from active_func import *
from cost_func import *
from common_func import *

class es_net(object):

    def __init__(self, layers, active_func=sigmoid, out_func=softmax):
        self.layers = layers
        self.weight = []
        self.bias = []
        self.grads_weight = [None] * self.layers
        self.grads_bias = [None] * self.layers
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
    
    def loss(self, input_val, test_label):
        y = self.forward_inference(input_x=input_val)
        return cross_entropy_err(output_val=y, label=test_label)
    
    def accuracy(self, input_val, test_label):
        y = self.forward_inference(input_x=input_val)
        y = np.argmax(y, axis=1)
        acc = np.sum(y == test_label) / float(input_val.shape[0])
        return acc
            
    def train(self, input_val, test_label, learning_rate=0.1):
        '''
        this lambda func has nothing to do with param W,
        self.loss() will use input_val, test_label as input
        and the self.loss() func depends on self.weight, self.bias
        the self.weight, self.bias param's delta value will have effect
        on the loss value when do forward_inference then affect the 
        numerical_gradient
        '''
        loss_W = lambda W: self.loss(input_val, test_label)

        for n in range(0, self.layers, 1):
            self.grads_weight[n] = numerical_gradient(loss_W, self.weight[n])
            self.grads_bias[n] = numerical_gradient(loss_W, self.bias[n])
        
        for n in range(0, self.layers, 1):
            self.weight[n] -= learning_rate * self.grads_weight[n]
            self.bias[n] -= learning_rate * self.grads_bias[n]
