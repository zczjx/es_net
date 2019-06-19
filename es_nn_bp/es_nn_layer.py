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
        self.W = weight
        self.b = bias
        self.x = None
        self.dW = None
        self.db = None
        self.name = 'affine_layer'
    
    def update_layer_nn_param(self, learning_rate=0.1):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def print_layer_name(self):
        print(self.name)

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.w) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T) 
        self.dW = np.dot(self.x.T, dout)
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

class conv_layer(object):
    def __init__(self, weight, bias, stride=1, pad=0):
        self.W = weight
        self.b = bias
        self.stride = stride
        self.pad = pad
        self.name = 'conv_layer'
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def update_layer_nn_param(self, learning_rate=0.1):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def print_layer_name(self):
        print(self.name)
    
    def forward(self, x):
        filter_num, channels, filter_height, filter_weight = self.W.shape
        in_num, channels, in_height, in_weight = x.shape
        out_height = 1 + int((in_height + 2*self.pad - filter_height) / self.stride)
        out_weight= 1 + int((in_weight + 2*self.pad - filter_weight) / self.stride)
        col = im2col(x, filter_height, filter_weight, self.stride, self.pad)
        col_W = self.W.reshape(filter_num, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(in_num, out_height, out_weight, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out 

    def backward(self, dout):
        filter_num, channels, filter_height, filter_weight = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, filter_num)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(filter_num, channels, filter_height, filter_weight)

        dcol = np.dot(dout, self,col_W.T)
        dx = col2im(dcol. self.x.shape, filter_height, filter_weight , self.stride, self.pad)

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

    

