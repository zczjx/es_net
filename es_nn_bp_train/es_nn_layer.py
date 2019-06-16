#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np
from active_func import *
from cost_func import *
from common_func import *

class ReLU_layer(object):
    def __init__(self):
        self.out_mask = None
        self.name = 'ReLU_layer'

    def update_layer_nn_param(self, learning_rate=0.1):
        pass

    def print_layer_name(self):
        print(self.name)
    
    def forward(self, x):
        self.out_mask = (x <= 0)
        out = x.copy()
        out[self.out_mask] = 0
        return out

    def backward(self, dout):
        dout[self.out_mask] = 0
        dx = dout
        return dx

class sigmoid_layer(object):
    def __init__(self):
        self.out = None
        self.name = 'sigmoid_layer'

    def update_layer_nn_param(self, learning_rate=0.1):
        pass
    
    def print_layer_name(self):
        print(self.name)

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class affine_layer(object):
    def __init__(self, weight, bias):
        self.w = weight
        self.b = bias
        self.x = None
        self.dw = None
        self.db = None
        self.name = 'affine_layer'
    
    def update_layer_nn_param(self, learning_rate=0.1):
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db

    def print_layer_name(self):
        print(self.name)

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.w.T) 
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class softmax_layer(object):
    def __init__(self):
        self.out = None
        self.name = 'softmax_layer'

    def update_layer_nn_param(self, learning_rate=0.1):
        pass

    def print_layer_name(self):
        print(self.name)
    
    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, label, dout=1):
        batch_size = label.shape[0]
        dx = self.out.copy()
        dx[np.arange(batch_size), label] -= 1
        dx = dx / batch_size
        return dx


if __name__=='__main__':
    layer = ReLU_layer()
    out = layer.forward(np.array([0, 1, -3, 5, 1, -9, 8, 7]))
    print(out)
    origin = [0, 1, -3, 5, 1, -9, 8, 7]
    origin.reverse()
    print('reverse: ', origin)
    origin.reverse()
    print('origin: ', origin)
    
    layer.print_layer_name()
    layer = softmax_layer()
    layer.print_layer_name()

    

