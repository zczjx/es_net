#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import sys, os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from active_func import *
from common_func import *
from es_net import *
from es_nn_layer import *
from mnist import load_mnist

if __name__=='__main__':
    (train_data, train_label), (test_data, test_label) = load_mnist(flatten=False)
    scale=0.01
    image_size = 28 * 28
    hidden_nodes = 100
    output_nodes = 10
    input_dim = (1, 28, 28)
    update_class = Momentum

    # random init the dnn param
    dnn = es_net()

    # Conv layer
    filter_num = 30
    filter_size = 5
    filter_pad = 0
    filter_stride = 1
    input_size = input_dim[1]
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

    scale = weight_init_scale(input_size=28*28, active_func='relu')
    dnn_weight_arr = scale * \
                    nd.random.normal(scale=0.01, 
                        shape=(filter_num, input_dim[0], filter_size, filter_size),
                        ctx=ctx)
    dnn_bias_arr = nd.zeros(filter_num, ctx=ctx)
    updater_obj = update_class(learning_rate=0.1)

    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, \
                    filter_pad, updater=updater_obj)
    dnn.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    dnn.add_layer(layer_obj=layer_tmp)

    # Pooling layer
    layer_tmp = pooling_layer(pool_h=2, pool_w=2, stride=2, pad=0)
    dnn.add_layer(layer_obj=layer_tmp)

    # Affine layer
    scale = weight_init_scale(pool_output_size, active_func='relu')
    dnn_weight_arr = scale * nd.random.normal(scale=0.01,
                                        shape=(pool_output_size, hidden_nodes),
                                        ctx=ctx)
    dnn_bias_arr = nd.zeros(hidden_nodes, ctx=ctx)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    dnn.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    dnn.add_layer(layer_obj=layer_tmp)

    # Affine layer
    scale = weight_init_scale(hidden_nodes, active_func='relu')
    dnn_weight_arr = scale * nd.random.normal(scale=0.01, 
                                        shape=(hidden_nodes, output_nodes),
                                        ctx=ctx)
    dnn_bias_arr = nd.zeros(output_nodes, ctx=ctx)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    dnn.add_layer(layer_obj=layer_tmp)

    # Softmax layer
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
    validate_batch_size = batch_size * 10

    for i in range(training_iters):
        sample_indices = np.random.choice(train_data_num, batch_size)
        train_data_batch = nd.array(train_data[sample_indices], ctx=ctx)
        train_label_batch = nd.array(train_label[sample_indices], ctx=ctx)

        print('start ', i, ' training iterations')
        dnn.train(train_data_batch, train_label_batch)
        loss_val = dnn.loss(train_data_batch, train_label_batch)
        train_loss_list.append(loss_val.asnumpy())

        if i % iter_per_epoch == 0:
        # if i % 30 == 0:
            sample_indices = np.random.choice(train_data_num, validate_batch_size)
            train_data_batch = nd.array(train_data[sample_indices], ctx=ctx)
            train_label_batch = nd.array(train_label[sample_indices], ctx=ctx)
            train_acc = dnn.accuracy(train_data_batch, train_label_batch)
            sample_indices = np.random.choice(test_data_num, validate_batch_size)
            test_data_batch = nd.array(test_data[sample_indices], ctx=ctx)
            test_label_batch = nd.array(test_label[sample_indices], ctx=ctx)
            test_acc = dnn.accuracy(test_data_batch, test_label_batch)
            train_acc_list.append(train_acc.asnumpy())
            test_acc_list.append(test_acc.asnumpy())
            print('finish epoch ', epochs_n)
            print("train acc, test acc | " + str(train_acc.asnumpy()) + ", " + str(test_acc.asnumpy()))
            epochs_n += 1
    
    # draw
    x = nd.arange(len(train_acc_list), ctx=ctx)
    plt.title('cnn training')
    plt.plot(x.asnumpy(), train_acc_list, label='train acc')
    plt.plot(x.asnumpy(), test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.show()

    






    
