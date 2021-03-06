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
    weight_init_std=0.01
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
    cnn_sgd = es_net()
    update_class = SGD

    # Conv layer

    dnn_weight_arr = weight_init_std * \
                    np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    dnn_bias_arr = np.zeros(filter_num)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, \
                    filter_pad, updater=updater_obj)
    cnn_sgd.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_sgd.add_layer(layer_obj=layer_tmp)

    # Pooling layer
    layer_tmp = pooling_layer(pool_h=2, pool_w=2, stride=2, pad=0)
    cnn_sgd.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(pool_output_size, hidden_nodes)
    dnn_bias_arr = np.zeros(hidden_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_sgd.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_sgd.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(hidden_nodes, output_nodes)
    dnn_bias_arr = np.zeros(output_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_sgd.add_layer(layer_obj=layer_tmp)

    # Softmax layer
    layer_tmp = softmax_layer()
    cnn_sgd.add_output_layer(layer_obj=layer_tmp)
    cnn_sgd.print_nn_layers()

    train_loss_sgd_list = []

    # cnn use Momentum

    cnn_momentum = es_net()
    update_class = Momentum

    # Conv layer

    dnn_weight_arr = weight_init_std * \
                    np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    dnn_bias_arr = np.zeros(filter_num)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, \
                    filter_pad, updater=updater_obj)
    cnn_momentum.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_momentum.add_layer(layer_obj=layer_tmp)

    # Pooling layer
    layer_tmp = pooling_layer(pool_h=2, pool_w=2, stride=2, pad=0)
    cnn_momentum.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(pool_output_size, hidden_nodes)
    dnn_bias_arr = np.zeros(hidden_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_momentum.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_momentum.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(hidden_nodes, output_nodes)
    dnn_bias_arr = np.zeros(output_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_momentum.add_layer(layer_obj=layer_tmp)

    # Softmax layer
    layer_tmp = softmax_layer()
    cnn_momentum.add_output_layer(layer_obj=layer_tmp)
    cnn_momentum.print_nn_layers()

    train_loss_momentum_list = []

    # cnn use AdaGrad

    cnn_adagrad = es_net()
    update_class = AdaGrad

    # Conv layer

    dnn_weight_arr = weight_init_std * \
                    np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    dnn_bias_arr = np.zeros(filter_num)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, \
                    filter_pad, updater=updater_obj)
    cnn_adagrad.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_adagrad.add_layer(layer_obj=layer_tmp)

    # Pooling layer
    layer_tmp = pooling_layer(pool_h=2, pool_w=2, stride=2, pad=0)
    cnn_adagrad.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(pool_output_size, hidden_nodes)
    dnn_bias_arr = np.zeros(hidden_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_adagrad.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_adagrad.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(hidden_nodes, output_nodes)
    dnn_bias_arr = np.zeros(output_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_adagrad.add_layer(layer_obj=layer_tmp)

    # Softmax layer
    layer_tmp = softmax_layer()
    cnn_adagrad.add_output_layer(layer_obj=layer_tmp)
    cnn_adagrad.print_nn_layers()

    train_loss_adagrad_list = []

    # cnn use adam

    cnn_adam = es_net()
    update_class = Adam

    # Conv layer

    dnn_weight_arr = weight_init_std * \
                    np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    dnn_bias_arr = np.zeros(filter_num)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = conv_layer(dnn_weight_arr, dnn_bias_arr, filter_stride, \
                    filter_pad, updater=updater_obj)
    cnn_adam.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_adam.add_layer(layer_obj=layer_tmp)

    # Pooling layer
    layer_tmp = pooling_layer(pool_h=2, pool_w=2, stride=2, pad=0)
    cnn_adam.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(pool_output_size, hidden_nodes)
    dnn_bias_arr = np.zeros(hidden_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_adam.add_layer(layer_obj=layer_tmp)

    # ReLU layer
    layer_tmp = ReLU_layer()
    cnn_adam.add_layer(layer_obj=layer_tmp)

    # Affine layer
    dnn_weight_arr = weight_init_std * np.random.randn(hidden_nodes, output_nodes)
    dnn_bias_arr = np.zeros(output_nodes)
    updater_obj = update_class(learning_rate=0.1)
    layer_tmp = affine_layer(weight=dnn_weight_arr, bias=dnn_bias_arr, \
                    updater=updater_obj)
    cnn_adam.add_layer(layer_obj=layer_tmp)

    # Softmax layer
    layer_tmp = softmax_layer()
    cnn_adam.add_output_layer(layer_obj=layer_tmp)
    cnn_adam.print_nn_layers()

    train_loss_adam_list = []

    



    train_data_num = train_data.shape[0]
    test_data_num = test_data.shape[0]
    batch_size = 100
    iter_per_epoch = max(train_data_num / batch_size, 1)
    training_iters = 121


    for i in range(training_iters):
        sample_indices = np.random.choice(train_data_num, batch_size)
        train_data_batch = train_data[sample_indices]
        train_label_batch = train_label[sample_indices]

        print('start ', i, ' training iterations')

        cnn_sgd.train(train_data_batch, train_label_batch)
        loss_val = cnn_sgd.loss(train_data_batch, train_label_batch)
        train_loss_sgd_list.append(loss_val)

        cnn_momentum.train(train_data_batch, train_label_batch)
        loss_val = cnn_momentum.loss(train_data_batch, train_label_batch)
        train_loss_momentum_list.append(loss_val)

        cnn_adagrad.train(train_data_batch, train_label_batch)
        loss_val = cnn_adagrad.loss(train_data_batch, train_label_batch)
        train_loss_adagrad_list.append(loss_val)

        cnn_adam.train(train_data_batch, train_label_batch)
        loss_val = cnn_adam.loss(train_data_batch, train_label_batch)
        train_loss_adam_list.append(loss_val)
    
    # draw
    x = np.arange(len(train_loss_sgd_list))
    plt.plot(x, train_loss_sgd_list, label='SGD')
    plt.plot(x, train_loss_momentum_list, label='Momentum')
    plt.plot(x, train_loss_adagrad_list, label='AdaGrad')
    plt.plot(x, train_loss_adam_list, label='Adam')
    plt.title('cnn training loss')
    plt.xlabel("iterations")
    plt.ylabel("loss val")
    plt.legend(loc='higher right')
    plt.show()
    






    
