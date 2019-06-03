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

    batch_size = 100
    for i in range(0, len(test_data), batch_size):
        y_batch = dnn.forward_inference(test_data[i:i+batch_size])
        result_batch = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(result_batch == test_label[i:i+batch_size])
        
    print("Final Accuracy: " + str(float(accuracy_cnt) / len(test_data)))
    
