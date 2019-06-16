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
    weight_init_std=0.01
    image_size = 28 * 28
    hidden_nodes = 30
    output_nodes = 10

    # random init the dnn param
    dnn = es_net(layers=2, active_func=ReLU, out_func=softmax)

    dnn_weight_arr = weight_init_std * np.random.randn(image_size, hidden_nodes)
    dnn_bias_arr = np.zeros(hidden_nodes)
    dnn.setup_nn_param(0, weight_arr = dnn_weight_arr, bias_arr = dnn_bias_arr)

    dnn_weight_arr = weight_init_std * np.random.randn(hidden_nodes, output_nodes)
    dnn_bias_arr = np.zeros(output_nodes)
    dnn.setup_nn_param(1, weight_arr = dnn_weight_arr, bias_arr = dnn_bias_arr)

    train_data_num = train_data.shape[0]
    batch_size = 100
    iter_per_epoch = max(train_data_num / batch_size, 1)
    training_iters = 1001

    print('train_data_num: ', train_data_num)
    print('iter_per_epoch: ', iter_per_epoch)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for i in range(training_iters):
        sample_indices = np.random.choice(train_data_num, batch_size)
        train_data_batch = train_data[sample_indices]
        train_label_batch = train_label[sample_indices]

        print('start ', i, ' training iterations')
        dnn.train(train_data_batch, train_label_batch)
        loss_val = dnn.loss(train_data_batch, train_label_batch)
        train_loss_list.append(loss_val)

        if i % 20 == 0:
            train_acc = dnn.accuracy(train_data, train_label)
            test_acc = dnn.accuracy(test_data, test_label)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    # draw

    # markers = {'train': 'o', 'test': 's'}
    # x = np.arange(len(train_loss_list))
    x = np.arange(len(train_acc_list))
    # plt.plot(x, train_loss_list, label='train acc')
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    # plt.xlabel("training_iters")
    # plt.ylabel("loss_val")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
   # plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()






    
