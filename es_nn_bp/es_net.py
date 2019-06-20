#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
from active_func import *
from cost_func import *
from common_func import *
from es_nn_layer import *
import pdb 

class es_net(object):

    def __init__(self):
        self.layers = []
        self.output_layer = None
        self.grads_weight = None
        self.grads_bias = None
        
    def add_layer(self, layer_obj):
        self.layers.append(layer_obj)
    
    def add_output_layer(self, layer_obj=softmax_layer()):
        self.output_layer = layer_obj
    
    def print_nn_layers(self):
        for layer_obj in self.layers:
            layer_obj.print_layer_name()
            print('|')
            print('|')
            print('v')
        self.output_layer.print_layer_name()

    def forward_inference(self, input_x):
        x = input_x
        for layer_obj in self.layers:
            x = layer_obj.forward(x)
        
        output = self.output_layer.forward(x)
        return output
    
    def loss(self, input_val, test_label):
        y = self.forward_inference(input_x=input_val)
        return cross_entropy_err(output_val=y, label=test_label)
    
    def accuracy(self, input_val, test_label, batch_size=100):
        acc = 0.0

        for i in range(int(input_val.shape[0] / batch_size)):
            x = input_val[i*batch_size:(i+1)*batch_size]
            t = test_label[i*batch_size:(i+1)*batch_size]
            y = self.forward_inference(input_x=x)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == t) 
        
        return acc / input_val.shape[0]
            
    def train(self, input_val, test_label, learning_rate=0.1):
        # forward
        loss_val = self.loss(input_val=input_val, test_label=test_label)
        #backward to calculate 
        dout = 1
        dout = self.output_layer.backward(label=test_label)
        self.layers.reverse()

        for layer_obj in self.layers:
            dout = layer_obj.backward(dout)
            
        self.layers.reverse()

        # update param
        for layer_obj in self.layers:
            dout = layer_obj.update_layer_nn_param(learning_rate)

        self.output_layer.update_layer_nn_param(learning_rate)



        
