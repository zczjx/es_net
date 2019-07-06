#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys, os
import pickle
import matplotlib.pyplot as plt
from active_func import *
from common_func import *
import numpy as np
import random
from es_net import *
from es_nn_layer import *
from es_nn_updater import *
from mnist import load_mnist

if __name__=='__main__':
    (train_data, train_label), (test_data, test_label) = load_mnist()
    scale=0.01
    image_size = 28 * 28
    hidden_nodes = 100
    output_nodes = 10
    update_class = Momentum

    # random init the dnn param
    dnn = es_net()

    scale = weight_init_scale(input_size=image_size, active_func='relu')
    dnn_weight_arr = scale * nd.random.normal(scale=0.01, 
                                shape=(image_size, hidden_nodes), ctx=ctx)
    dnn_bias_arr = nd.zeros(hidden_nodes, ctx=ctx)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    dnn.add_layer(layer_obj=layer_tmp)

    layer_tmp = ReLU_layer()
    dnn.add_layer(layer_obj=layer_tmp)

    scale = weight_init_scale(input_size=hidden_nodes, active_func='relu')
    dnn_weight_arr = scale * nd.random.normal(scale=0.01,
                                shape=(hidden_nodes, output_nodes), ctx=ctx)
    dnn_bias_arr = nd.zeros(output_nodes, ctx=ctx)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    dnn.add_layer(layer_obj=layer_tmp)

    layer_tmp = softmax_layer()
    dnn.add_output_layer(layer_obj=layer_tmp)
    dnn.print_nn_layers()

    train_data_num = train_data.shape[0]
    test_data_num = test_data.shape[0]
    batch_size = 100
    iter_per_epoch = max(train_data_num / batch_size, 1)
    training_iters = 12001

    print('train_data_num: ', train_data_num)
    print('iter_per_epoch: ', iter_per_epoch)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_n = 0

    for i in range(training_iters):
        sample_indices = np.random.choice(train_data_num, batch_size)
        train_data_batch = nd.array(train_data[sample_indices], ctx=ctx)
        train_label_batch = nd.array(train_label[sample_indices], ctx=ctx)

        # print('start ', i, ' training iterations')
        dnn.train(train_data_batch, train_label_batch)
        loss_val = dnn.loss(train_data_batch, train_label_batch)
        train_loss_list.append(loss_val)

        if i % iter_per_epoch == 0:
            sample_indices =np.random.choice(train_data_num, batch_size*10)
            train_data_batch = nd.array(train_data[sample_indices], ctx=ctx)
            train_label_batch = nd.array(train_label[sample_indices], ctx=ctx)
            train_acc = dnn.accuracy(train_data_batch, train_label_batch, batch_size)
            sample_indices = np.random.choice(test_data_num, batch_size)
            test_data_batch = nd.array(test_data[sample_indices], ctx=ctx)
            test_label_batch = nd.array(test_label[sample_indices], ctx=ctx)
            test_acc = dnn.accuracy(test_data_batch, test_label_batch, batch_size)
            train_acc_list.append(train_acc.asnumpy())
            test_acc_list.append(test_acc.asnumpy())
            print('finish epoch ', epochs_n)
            print("train acc, test acc | " + str(train_acc.asnumpy()) + ", " + str(test_acc.asnumpy()))
            epochs_n += 1
    
    # draw
    x = nd.arange(len(train_acc_list), ctx=ctx)
    plt.title('fnn training')
    plt.plot(x.asnumpy(), train_acc_list, label='train acc')
    plt.plot(x.asnumpy(), test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.show()
    






    
