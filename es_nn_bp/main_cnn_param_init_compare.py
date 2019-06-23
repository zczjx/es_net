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

    # conv param
    filter_num = 30
    filter_size = 5
    filter_pad = 0
    filter_stride = 1
    input_size = input_dim[1]
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
    

    # cnn use SGD
    cnn_default = es_net()
    update_class = SGD

    # Conv layer
    scale = weight_init_scale(input_size=28*28, active_func='default')
    dnn_weight_arr = scale * \
                    np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    dnn_bias_arr = np.zeros(filter_num)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, \
                    filter_pad, updater=updater_obj)
    cnn_default.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_default.add_layer(layer_obj=layer_tmp)

    # Pooling layer
    layer_tmp = pooling_layer(pool_h=2, pool_w=2, stride=2, pad=0)
    cnn_default.add_layer(layer_obj=layer_tmp)

    # Affine layer
    scale = weight_init_scale(input_size=pool_output_size, active_func='default')
    dnn_weight_arr = scale * np.random.randn(pool_output_size, hidden_nodes)
    dnn_bias_arr = np.zeros(hidden_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_default.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_default.add_layer(layer_obj=layer_tmp)

    # Affine layer
    scale = weight_init_scale(input_size=hidden_nodes, active_func='default')
    dnn_weight_arr = scale * np.random.randn(hidden_nodes, output_nodes)
    dnn_bias_arr = np.zeros(output_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_default.add_layer(layer_obj=layer_tmp)

    # Softmax layer
    layer_tmp = softmax_layer()
    cnn_default.add_output_layer(layer_obj=layer_tmp)
    cnn_default.print_nn_layers()

    train_loss_default_list = []

    # cnn use Momentum

    cnn_he= es_net()
    update_class = SGD

    # Conv layer

    scale = weight_init_scale(input_size=28*28, active_func='relu')
    dnn_weight_arr = scale * \
                    np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    dnn_bias_arr = np.zeros(filter_num)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, \
                    filter_pad, updater=updater_obj)
    cnn_he.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_he.add_layer(layer_obj=layer_tmp)

    # Pooling layer
    layer_tmp = pooling_layer(pool_h=2, pool_w=2, stride=2, pad=0)
    cnn_he.add_layer(layer_obj=layer_tmp)

    # Affine layer
    scale = weight_init_scale(pool_output_size, active_func='relu')
    dnn_weight_arr = scale * np.random.randn(pool_output_size, hidden_nodes)
    dnn_bias_arr = np.zeros(hidden_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_he.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_he.add_layer(layer_obj=layer_tmp)

    # Affine layer
    scale = weight_init_scale(hidden_nodes, active_func='relu')
    dnn_weight_arr = scale * np.random.randn(hidden_nodes, output_nodes)
    dnn_bias_arr = np.zeros(output_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_he.add_layer(layer_obj=layer_tmp)

    # Softmax layer
    layer_tmp = softmax_layer()
    cnn_he.add_output_layer(layer_obj=layer_tmp)
    cnn_he.print_nn_layers()

    train_loss_he_list = []

    train_data_num = train_data.shape[0]
    test_data_num = test_data.shape[0]
    batch_size = 100
    iter_per_epoch = max(train_data_num / batch_size, 1)
    training_iters = 1201


    for i in range(training_iters):
        sample_indices = np.random.choice(train_data_num, batch_size)
        train_data_batch = train_data[sample_indices]
        train_label_batch = train_label[sample_indices]

        print('start ', i, ' training iterations')

        cnn_default.train(train_data_batch, train_label_batch)
        loss_val = cnn_default.loss(train_data_batch, train_label_batch)
        train_loss_default_list.append(loss_val)

        cnn_he.train(train_data_batch, train_label_batch)
        loss_val = cnn_he.loss(train_data_batch, train_label_batch)
        train_loss_he_list.append(loss_val)

    # draw
    x = np.arange(len(train_loss_default_list))
    plt.plot(x, train_loss_default_list, label='default')
    plt.plot(x, train_loss_he_list, label='He')
    plt.title('cnn training loss')
    plt.xlabel("iterations")
    plt.ylabel("loss val")
    plt.legend(loc='higher right')
    plt.show()
    