#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from active_func import *
from common_func import *
from es_net import *
from mnist import load_mnist

if __name__=='__main__':
    (train_data, train_label), (test_data, test_label) = load_mnist()
    with open("sample_weight.pkl", 'rb') as f:
        nn_param = pickle.load(f)
    accuracy_cnt = 0
    dnn = es_net(layers=3, active_func=sigmoid, out_func=softmax)
    dnn.setup_nn_param(0, weight_arr = nn_param['W1'], bias_arr = nn_param['b1'])
    dnn.setup_nn_param(1, weight_arr = nn_param['W2'], bias_arr = nn_param['b2'])
    dnn.setup_nn_param(2, weight_arr = nn_param['W3'], bias_arr = nn_param['b3'])

    for i in range(len(test_data)):
        y = dnn.forward_inference(test_data[i])
        result = np.argmax(y)
        if result == test_label[i]:
            accuracy_cnt += 1
        else:
            print('failed img idx: ', i, 'err val: ', result, 
                'correct val: ', test_label[i])
 
    print("Final Accuracy: " + str(float(accuracy_cnt) / len(test_data)))
    
